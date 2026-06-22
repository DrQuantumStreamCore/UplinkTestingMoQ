#!/usr/bin/env python
"""
pcap -> StreamCore dataset bridge.

Turns real packet captures into the data StreamCore consumes. Two levels:

  --emit moq-trace
      A single capture -> per-window offered-Mbps array for
      tools/moq_uplink_test.py (drives the uplink test with REAL traffic shape).

  --emit dataset
      Writes a VNF dataset (input_dataset.pkl / output_dataset.pkl).
        * single capture  -> input features are real; output is a clearly
          marked PASSTHROUGH placeholder (no transfer function).
        * --egress-pcap   -> REAL per-VNF transfer function: ingress and egress
          captures are correlated per packet (payload hash, which a forwarding
          VNF preserves) to measure served throughput, latency (time_in_sys)
          and loss. Tag the capture with the resource the VNF had via --res;
          run several captures at different --res values and concatenate to
          build a training set the slice model can learn resource sensitivity
          from.

Uplink direction: --uplink-src CIDR/host (the contributor/device side); with no
filter the dominant byte sender is assumed to be the uplink publisher.

Usage:
    # offered-load trace
    python tools/pcap_to_dataset.py ingress.pcap --uplink-src 10.0.0.0/24 \
        --emit moq-trace --out offered.npy

    # REAL transfer-function dataset for the RAN at 1500 millicores
    python tools/pcap_to_dataset.py ingress.pcap --egress-pcap egress.pcap \
        --uplink-src 10.0.0.0/24 --emit dataset --vnf ran --res 1500 \
        --out-dir net_model_dataset
"""
import argparse, collections, hashlib, ipaddress, os, sys
import numpy as np
import pandas as pd

try:
    from scapy.all import PcapReader, IP, IPv6
except Exception as e:  # pragma: no cover
    print("scapy is required: pip install scapy  (", e, ")")
    sys.exit(2)


def _in_net(addr, net):
    try:
        return ipaddress.ip_address(addr) in net
    except ValueError:
        return False


def _pkt_key(l3):
    """Per-packet identity preserved by a forwarding VNF: hash of the L4 payload
    (UDP/TCP header + data). Falls back to header fields if there is no payload."""
    payload = bytes(l3.payload)
    if payload:
        return hashlib.blake2b(payload, digest_size=8).digest()
    return (getattr(l3, "id", 0), str(l3.src), str(l3.dst), len(l3))


def collect(path, uplink_src=None):
    """Return time-sorted (times, lens, keys) for uplink-direction packets."""
    net = ipaddress.ip_network(uplink_src, strict=False) if uplink_src else None
    recs, byte_by_src = [], {}
    with PcapReader(path) as pr:
        for pkt in pr:
            l3 = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if l3 is None:
                continue
            recs.append((float(pkt.time), str(l3.src), int(len(pkt)), _pkt_key(l3)))
            byte_by_src[str(l3.src)] = byte_by_src.get(str(l3.src), 0) + int(len(pkt))
    if not recs:
        raise SystemExit(f"No IP packets in {path}")
    if net is not None:
        recs = [r for r in recs if _in_net(r[1], net)]
    else:
        top = max(byte_by_src, key=byte_by_src.get)
        print(f"[info] {os.path.basename(path)}: no --uplink-src; uplink publisher = {top}")
        recs = [r for r in recs if r[1] == top]
    if not recs:
        raise SystemExit(f"No uplink packets after direction filter in {path}")
    recs.sort(key=lambda r: r[0])
    times = np.array([r[0] for r in recs])
    lens = np.array([r[2] for r in recs], dtype=float)
    keys = [r[3] for r in recs]
    return times, lens, keys


# --- single-capture (offered load) -------------------------------------------
def windowize(times, lens, window_s):
    t0 = times.min()
    bins = ((times - t0) / window_s).astype(int)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        wl, wt = lens[m], np.sort(times[m])
        if len(wl) < 2:
            continue
        iat = np.diff(wt)
        rows.append(dict(
            packet_size=float(wl.mean()), packet_rate=float(len(wl) / window_s),
            throughput=float(wl.sum() / window_s),
            inter_arrival_time_mean=float(iat.mean()),
            inter_arrival_time_std=float(iat.std()),
            time_stamp_arr=[0.0, float(iat.mean())]))
    df = pd.DataFrame(rows)
    print(f"[info] {len(df)} windows; mean offered {df.throughput.mean()*8/1e6:.2f} Mbps, "
          f"peak {df.throughput.max()*8/1e6:.2f} Mbps")
    return df


def emit_moq_trace(df, out):
    mbps = (df["throughput"].values * 8 / 1e6).astype(float)
    np.save(out, mbps)
    print(f"[ok] wrote {len(mbps)} per-window offered-Mbps samples -> {out}")


# --- ingress<->egress correlation (real transfer function) -------------------
def correlate(ing, egr, slack_s=1e-3):
    """Greedy per-packet match by key; returns DataFrame with per-ingress-packet
    matched flag, egress length and latency."""
    it, il, ik = ing
    et, el, ek = egr
    idx = collections.defaultdict(collections.deque)
    for t, l, k in sorted(zip(et, el, ek), key=lambda x: x[0]):
        idx[k].append((t, l))
    out = []
    for t, l, k in zip(it, il, ik):
        dq = idx.get(k)
        hit = None
        while dq:
            te, le = dq[0]
            if te >= t - slack_s:           # egress can't precede ingress
                hit = (te, le); dq.popleft(); break
            dq.popleft()                    # discard stale earlier egress
        out.append((t, l, hit is not None,
                    (hit[0] - t) if hit else np.nan,
                    hit[1] if hit else 0.0))
    m = pd.DataFrame(out, columns=["t_in", "len_in", "matched", "latency", "len_out"])
    loss = 1.0 - m.matched.mean()
    print(f"[info] correlated {int(m.matched.sum())}/{len(m)} packets "
          f"(loss {loss:.1%}, median latency {np.nanmedian(m.latency)*1e3:.1f} ms)")
    return m


def transfer_rows(m, window_s):
    """Per-window input (ingress) + output (served egress) features + time_in_sys."""
    t0 = m.t_in.min()
    m = m.assign(bin=((m.t_in - t0) / window_s).astype(int))
    rows = []
    for _, g in m.groupby("bin"):
        if len(g) < 2:
            continue
        served = g[g.matched]
        in_iat = np.diff(np.sort(g.t_in.values))
        if len(served) >= 2:
            out_t = np.sort(served.t_in.values + served.latency.values)
            out_iat = np.diff(out_t)
            out_ps, out_rate = served.len_out.mean(), len(served) / window_s
            tis = float(served.latency.mean())
        else:
            out_iat = np.array([0.0]); out_ps = 0.0; out_rate = 0.0; tis = window_s
        rows.append(dict(
            in_packet_size=float(g.len_in.mean()), in_packet_rate=float(len(g) / window_s),
            in_throughput=float(g.len_in.sum() / window_s),
            in_iat_mean=float(in_iat.mean()), in_iat_std=float(in_iat.std()),
            out_packet_size=float(out_ps), out_packet_rate=float(out_rate),
            out_throughput=float(served.len_out.sum() / window_s),
            out_iat_mean=float(out_iat.mean()), out_iat_std=float(out_iat.std()),
            time_in_sys=float(tis)))
    return pd.DataFrame(rows)


def write_dataset(inp_df, out_df, vnf, out_dir):
    d = os.path.join(out_dir, vnf)
    os.makedirs(d, exist_ok=True)
    inp_df = inp_df[["packet_size", "packet_rate", "throughput",
                     "inter_arrival_time_mean", "inter_arrival_time_std", "res",
                     "time_stamp_arr"]]
    out_df = out_df[["packet_size", "packet_rate", "throughput",
                     "inter_arrival_time_mean", "inter_arrival_time_std", "time_in_sys",
                     "time_stamp_arr", "time_in_sys_arr"]]
    inp_df.to_pickle(os.path.join(d, "input_dataset.pkl"))
    out_df.to_pickle(os.path.join(d, "output_dataset.pkl"))
    print(f"[ok] wrote {len(inp_df)} rows -> {d}/(input|output)_dataset.pkl")


def emit_dataset(args):
    ing = collect(args.pcap, args.uplink_src)
    if args.egress_pcap:
        egr = collect(args.egress_pcap, args.uplink_src)
        m = correlate(ing, egr, slack_s=args.window_ms / 1e3)
        tf = transfer_rows(m, args.window_ms / 1e3)
        if tf.empty:
            raise SystemExit("No windows with >=2 packets; capture too short.")
        inp = pd.DataFrame(dict(
            packet_size=tf.in_packet_size, packet_rate=tf.in_packet_rate,
            throughput=tf.in_throughput, inter_arrival_time_mean=tf.in_iat_mean,
            inter_arrival_time_std=tf.in_iat_std, res=float(args.res),
            time_stamp_arr=[[0.0, float(x)] for x in tf.in_iat_mean]))
        out = pd.DataFrame(dict(
            packet_size=tf.out_packet_size, packet_rate=tf.out_packet_rate,
            throughput=tf.out_throughput, inter_arrival_time_mean=tf.out_iat_mean,
            inter_arrival_time_std=tf.out_iat_std, time_in_sys=tf.time_in_sys,
            time_stamp_arr=[[0.0, float(x)] for x in tf.out_iat_mean],
            time_in_sys_arr=[[float(x)] for x in tf.time_in_sys]))
        served = out.throughput.sum() / inp.throughput.sum()
        print(f"[info] REAL transfer function: served/offered = {served:.1%}, "
              f"mean time_in_sys = {out.time_in_sys.mean()*1e3:.1f} ms, res = {args.res}")
        write_dataset(inp, out, args.vnf, args.out_dir)
    else:
        df = windowize(*ing[:2], args.window_ms / 1e3)
        print("[warn] no --egress-pcap: output is a PASSTHROUGH placeholder "
              "(output==input); resource/served labels are NOT real.")
        inp = df.copy(); inp["res"] = float(args.res)
        out = df.copy()
        out["time_in_sys"] = 0.005
        out["time_stamp_arr"] = [[0.0, float(x)] for x in out.inter_arrival_time_mean]
        out["time_in_sys_arr"] = [[0.005] for _ in range(len(out))]
        write_dataset(inp, out, args.vnf, args.out_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("--uplink-src", default=None, help="CIDR/host of the uplink sender")
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--emit", choices=["moq-trace", "dataset"], default="moq-trace")
    ap.add_argument("--out", default="moq_offered.npy")
    ap.add_argument("--vnf", default="ran", choices=["ran", "ovs", "upf"])
    ap.add_argument("--out-dir", default="net_model_dataset")
    ap.add_argument("--res", type=float, default=1500.0,
                    help="resource value the VNF had during this capture (transfer-function label)")
    ap.add_argument("--egress-pcap", default=None,
                    help="egress capture; enables real ingress<->egress correlation")
    args = ap.parse_args()

    if args.emit == "moq-trace":
        t, l, _ = collect(args.pcap, args.uplink_src)
        emit_moq_trace(windowize(t, l, args.window_ms / 1e3), args.out)
    else:
        emit_dataset(args)


if __name__ == "__main__":
    main()
