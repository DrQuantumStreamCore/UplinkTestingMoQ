#!/usr/bin/env python
"""
REAL MoQ-style uplink transmission over live QUIC (aioquic), on loopback.

Unlike tools/moq_uplink_test.py (a pure traffic model), this actually opens a
QUIC connection and pushes media "objects" UPLINK from a publisher to a relay,
one unidirectional stream per object, with MoQ-like group cadence (a large key
object every GoP, then smaller delta objects). The receiver timestamps every
object, so the measured throughput reflects real QUIC framing, pacing and
congestion control over the socket.

Output: a per-window offered-Mbps trace (.npy) measured at the receiver, which
feeds tools/moq_uplink_test.py / StreamCore exactly like a testbed capture would
(no sudo / pcap needed; it's an app-level measurement of a real transfer).

Run:
    python tools/moq_quic_transmit.py --seconds 2 --target-mbps 12 --out moq_real.npy
"""
import argparse, asyncio, datetime, ssl, time
import numpy as np

from aioquic.asyncio import connect, serve
from aioquic.asyncio.protocol import QuicConnectionProtocol
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import StreamDataReceived

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _self_signed():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime(2020, 1, 1))
            .not_valid_after(datetime.datetime(2030, 1, 1))
            .sign(key, hashes.SHA256()))
    return cert, key


class Receiver(QuicConnectionProtocol):
    """Relay side: record (recv_time, nbytes) per completed object stream."""
    instances = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        Receiver.instances.append(self)
        self.records = []
        self._acc = {}

    def quic_event_received(self, event):
        if isinstance(event, StreamDataReceived):
            sid = event.stream_id
            self._acc[sid] = self._acc.get(sid, 0) + len(event.data)
            if event.end_stream:
                self.records.append((time.time(), self._acc.pop(sid, 0)))


def moq_schedule(seconds, target_mbps, fps=30, gop=30, key_ratio=6.0):
    n = fps * seconds
    avg_bytes = target_mbps * 1e6 / 8 / fps
    sizes = np.array([key_ratio if i % gop == 0 else 1.0 for i in range(n)])
    sizes *= avg_bytes / sizes.mean()
    return np.clip(sizes, 64, 1_000_000).astype(int), 1.0 / fps


async def run(host, port, seconds, target_mbps, out, window_ms):
    cert, key = _self_signed()
    server_cfg = QuicConfiguration(is_client=False, max_datagram_frame_size=65536)
    server_cfg.certificate = cert
    server_cfg.private_key = key
    client_cfg = QuicConfiguration(is_client=True, max_datagram_frame_size=65536)
    client_cfg.verify_mode = ssl.CERT_NONE

    server = await serve(host, port, configuration=server_cfg, create_protocol=Receiver)

    sizes, interval = moq_schedule(seconds, target_mbps)
    print(f"[tx] {len(sizes)} objects, target {target_mbps} Mbps, "
          f"key/delta sizes {sizes.max()}/{int(np.median(sizes))} B")

    async with connect(host, port, configuration=client_cfg) as client:
        await client.wait_connected()
        payloads = {s: bytes(int(s)) for s in np.unique(sizes)}
        t_start = time.time()
        for s in sizes:
            sid = client._quic.get_next_available_stream_id(is_unidirectional=True)
            client._quic.send_stream_data(sid, payloads[s], end_stream=True)
            client.transmit()
            await asyncio.sleep(interval)
        await asyncio.sleep(0.7)  # let the tail drain
        t_end = time.time()

    server.close()

    recs = []
    for inst in Receiver.instances:
        recs.extend(inst.records)
    if not recs:
        raise SystemExit("No objects received — transmission failed.")
    times = np.array([r[0] for r in recs]) - t_start
    lens = np.array([r[1] for r in recs], dtype=float)
    total_mb = lens.sum() * 8 / 1e6
    wall = t_end - t_start
    print(f"[rx] received {len(recs)} objects, {lens.sum()/1e3:.0f} kB, "
          f"mean rate {total_mb/wall:.2f} Mbps over {wall:.2f}s")

    # window the receiver trace into per-window Mbps
    w = window_ms / 1e3
    nb = int(np.ceil(times.max() / w)) + 1
    binned = np.zeros(nb)
    for t, l in zip(times, lens):
        binned[int(t / w)] += l
    mbps = binned * 8 / 1e6 / w
    mbps = mbps[mbps > 0]
    np.save(out, mbps)
    print(f"[ok] wrote {len(mbps)} per-window Mbps samples -> {out}  "
          f"(mean {mbps.mean():.2f}, peak {mbps.max():.2f} Mbps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=4455)
    ap.add_argument("--seconds", type=int, default=2)
    ap.add_argument("--target-mbps", type=float, default=12.0)
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--out", default="moq_real.npy")
    args = ap.parse_args()
    asyncio.run(run(args.host, args.port, args.seconds, args.target_mbps,
                    args.out, args.window_ms))


if __name__ == "__main__":
    main()
