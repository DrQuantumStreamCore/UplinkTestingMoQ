#!/usr/bin/env python
"""
pcap -> StreamCore traffic-profile dataset bridge.

Turns a real packet capture (testbed / MoQ uplink contributor) into the input
traffic profiles StreamCore consumes: windowed packet_size, packet_rate,
throughput, inter-arrival statistics. This replaces the *synthetic* offered-load
model with measured traffic shape.

SCOPE / HONESTY
---------------
A pcap gives you the OFFERED LOAD (input features). It does NOT give you a VNF
transfer function (resource -> served throughput / latency); that still needs
testbed instrumentation (vary the VNF resource, measure egress). So:
  * --emit moq-trace : per-window offered Mbps array for tools/moq_uplink_test.py
                       (drives the uplink test with REAL traffic shape).
  * --emit dataset   : writes input_dataset.pkl for a VNF + a PASSTHROUGH
                       output_dataset.pkl (output==input) clearly marked as a
                       placeholder; supply --egress-pcap to measure real output.

Uplink direction: pass --uplink-src CIDR/host (the contributor/device side) so
packets sourced there are counted as uplink. With no filter, the dominant byte
sender is assumed to be the uplink publisher.

Usage:
    python tools/pcap_to_dataset.py capture.pcap --uplink-src 10.0.0.0/24 \
        --window-ms 100 --emit moq-trace --out moq_offered.npy
    python tools/pcap_to_dataset.py capture.pcap --emit dataset --vnf ran \
        --out-dir net_model_dataset
"""
import argparse, ipaddress, os, sys
import numpy as np
import pandas as pd

try:
    from scapy.all import PcapReader, IP, IPv6
except Exception as e:  # pragma: no cover
    print("scapy is required: pip install scapy  (", e, ")")
    sys.exit(2)

MBPS_TO_BYTES = 1e6 / 8.0


def _in_net(addr, net):
    if net is None:
        return None
    try:
        return ipaddress.ip_address(addr) in net
    except ValueError:
        return False


def read_packets(path, uplink_src=None):
    """Yield (timestamp, src, dst, length) for IP/IPv6 packets."""
    net = ipaddress.ip_network(uplink_src, strict=False) if uplink_src else None
    with PcapReader(path) as pr:
        for pkt in pr:
            l3 = pkt.getlayer(IP) or pkt.getlayer(IPv6)
            if l3 is None:
                continue
            yield float(pkt.time), l3.src, l3.dst, int(len(pkt)), net


def collect(path, uplink_src=None):
    times, srcs, lens = [], [], []
    net = None
    byte_by_src = {}
    for t, s, d, ln, net in read_packets(path, uplink_src):
        times.append(t); srcs.append(s); lens.append(ln)
        byte_by_src[s] = byte_by_src.get(s, 0) + ln
    if not times:
        raise SystemExit("No IP packets found in capture.")
    times = np.array(times); lens = np.array(lens, dtype=float)
    # uplink mask
    if net is not None:
        mask = np.array([_in_net(s, net) for s in srcs], dtype=bool)
    else:
        top = max(byte_by_src, key=byte_by_src.get)
        mask = np.array([s == top for s in srcs], dtype=bool)
        print(f"[info] no --uplink-src; assuming uplink publisher = {top}")
    return times[mask], lens[mask]


def windowize(times, lens, window_s):
    """Aggregate packets into fixed time windows -> per-window features."""
    if len(times) == 0:
        raise SystemExit("No uplink packets after direction filtering.")
    t0 = times.min()
    bins = ((times - t0) / window_s).astype(int)
    rows = []
    for b in np.unique(bins):
        m = bins == b
        wl = lens[m]; wt = np.sort(times[m])
        if len(wl) < 2:
            continue
        dur = window_s
        thr_bytes = wl.sum() / dur                      # bytes/s
        pkt_rate = len(wl) / dur                         # packets/s
        iat = np.diff(wt)
        rows.append(dict(
            packet_size=float(wl.mean()),
            packet_rate=float(pkt_rate),
            throughput=float(thr_bytes),
            inter_arrival_time_mean=float(iat.mean()),
            inter_arrival_time_std=float(iat.std()),
            time_stamp_arr=[0.0, float(iat.mean())],
        ))
    df = pd.DataFrame(rows)
    print(f"[info] {len(df)} windows of {window_s*1e3:.0f} ms; "
          f"mean offered {df.throughput.mean()*8/1e6:.2f} Mbps, "
          f"peak {df.throughput.max()*8/1e6:.2f} Mbps")
    return df


def emit_moq_trace(df, out):
    mbps = (df["throughput"].values * 8 / 1e6).astype(float)
    np.save(out, mbps)
    print(f"[ok] wrote {len(mbps)} per-window offered-Mbps samples -> {out}")
    print(f"     load in moq_uplink_test via: np.load('{out}')")


def emit_dataset(df, vnf, out_dir, res_value, egress_df=None):
    d = os.path.join(out_dir, vnf)
    os.makedirs(d, exist_ok=True)
    inp = df.copy()
    inp["res"] = float(res_value)
    inp = inp[["packet_size", "packet_rate", "throughput",
               "inter_arrival_time_mean", "inter_arrival_time_std", "res",
               "time_stamp_arr"]]
    # Output: measured egress if provided, else PASSTHROUGH placeholder.
    src = egress_df if egress_df is not None else df
    out = src.copy()
    if egress_df is None:
        print("[warn] no --egress-pcap: output_dataset is a PASSTHROUGH placeholder "
              "(output==input). Resource/served-throughput labels are NOT real.")
    out["time_in_sys"] = 0.005
    out["time_stamp_arr"] = [[0.0, float(m)] for m in out["inter_arrival_time_mean"]]
    out["time_in_sys_arr"] = [[0.005] for _ in range(len(out))]
    out = out[["packet_size", "packet_rate", "throughput",
               "inter_arrival_time_mean", "inter_arrival_time_std", "time_in_sys",
               "time_stamp_arr", "time_in_sys_arr"]]
    inp.to_pickle(os.path.join(d, "input_dataset.pkl"))
    out.to_pickle(os.path.join(d, "output_dataset.pkl"))
    print(f"[ok] wrote {len(inp)} rows -> {d}/(input|output)_dataset.pkl")


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
    ap.add_argument("--res", type=float, default=1500.0, help="placeholder resource value")
    ap.add_argument("--egress-pcap", default=None, help="optional egress capture for real output labels")
    args = ap.parse_args()

    times, lens = collect(args.pcap, args.uplink_src)
    df = windowize(times, lens, args.window_ms / 1e3)
    egress_df = None
    if args.egress_pcap:
        et, el = collect(args.egress_pcap, args.uplink_src)
        egress_df = windowize(et, el, args.window_ms / 1e3)

    if args.emit == "moq-trace":
        emit_moq_trace(df, args.out)
    else:
        emit_dataset(df, args.vnf, args.out_dir, args.res, egress_df)


if __name__ == "__main__":
    main()
