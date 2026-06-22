#!/usr/bin/env python
"""
Generate a synthetic, schema-correct dataset for StreamCore so the pipeline is
runnable end-to-end (the published repo ships no data).

This is a SMOKE / INTEGRATION dataset, not a validation dataset: output
throughput is a saturating function of input throughput and the VNF resource,
so the slice model can learn a resource->throughput sensitivity and the
allocator has a real surface to optimise. It does NOT reproduce real testbed
measurements.

Schema (reverse-engineered from data_generator.py / slice_model.py):

  input_dataset.pkl  columns (order matters after time_stamp_arr is dropped):
    packet_size, packet_rate, throughput(BYTES/s), inter_arrival_time_mean,
    inter_arrival_time_std, res, time_stamp_arr
  output_dataset.pkl columns (order matters after *_arr are dropped):
    packet_size, packet_rate, throughput(BYTES/s), inter_arrival_time_mean,
    inter_arrival_time_std, time_in_sys, time_stamp_arr, time_in_sys_arr

throughput is stored in BYTES/s because DataGenerator multiplies it by 8/1e6.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
MBPS_TO_BYTES = 1e6 / 8.0

# Per-VNF capacity model: usable capacity (Mbps) as a function of the resource.
# RAN/UPF res = CPU millicores; OvS res = transport bandwidth (Mbps).
VNF_CFG = {
    "ran": dict(res_lo=500.0,  res_hi=3000.0, cap=lambda r: 0.02 * r,        loss=0.03),
    "ovs": dict(res_lo=5.0,    res_hi=60.0,   cap=lambda r: 1.00 * r,        loss=0.01),
    "upf": dict(res_lo=200.0,  res_hi=200.0,  cap=lambda r: 0.40 * r + 5.0,  loss=0.01),
}
TP_LO, TP_HI = 5.0, 60.0          # input throughput range (Mbps)
N_ROWS = 3000


def _make_vnf(vnf, n=N_ROWS):
    cfg = VNF_CFG[vnf]
    in_tp = RNG.uniform(TP_LO, TP_HI, n)                       # Mbps
    res = RNG.uniform(cfg["res_lo"], cfg["res_hi"], n)
    cap = cfg["cap"](res)                                      # Mbps
    # Saturating service: limited by min(offered, capacity), minus a small loss,
    # with a soft knee so the surface is smooth/differentiable-friendly.
    served = np.minimum(in_tp, cap)
    knee = in_tp * (1.0 - np.exp(-cap / np.maximum(in_tp, 1e-6)))
    out_tp = (0.5 * served + 0.5 * knee) * (1.0 - cfg["loss"])
    out_tp = np.clip(out_tp + RNG.normal(0, 0.3, n), 0.1, in_tp)

    pkt_size = RNG.normal(1200, 120, n).clip(400, 1500)        # bytes
    in_bytes = in_tp * MBPS_TO_BYTES
    out_bytes = out_tp * MBPS_TO_BYTES
    in_rate = in_bytes / pkt_size                             # packets/s
    out_rate = out_bytes / pkt_size
    iat_mean = 1.0 / np.maximum(in_rate, 1e-6)
    iat_std = iat_mean * RNG.uniform(0.1, 0.4, n)
    out_iat_mean = 1.0 / np.maximum(out_rate, 1e-6)
    out_iat_std = out_iat_mean * RNG.uniform(0.1, 0.4, n)
    # queueing-style latency: grows as offered load approaches capacity
    util = np.clip(in_tp / np.maximum(cap, 1e-6), 0, 0.999)
    time_in_sys = (0.001 + 0.02 * util / (1.0 - util)).clip(0.001, 0.5)

    res_col = 200.0 if vnf == "upf" else res

    inp = pd.DataFrame({
        "packet_size": pkt_size,
        "packet_rate": in_rate,
        "throughput": in_bytes,
        "inter_arrival_time_mean": iat_mean,
        "inter_arrival_time_std": iat_std,
        "res": res_col,
        "time_stamp_arr": [[0.0, float(m)] for m in iat_mean],
    })
    out = pd.DataFrame({
        "packet_size": pkt_size,
        "packet_rate": out_rate,
        "throughput": out_bytes,
        "inter_arrival_time_mean": out_iat_mean,
        "inter_arrival_time_std": out_iat_std,
        "time_in_sys": time_in_sys,
        "time_stamp_arr": [[0.0, float(m)] for m in out_iat_mean],
        "time_in_sys_arr": [[float(t)] for t in time_in_sys],
    })
    return inp, out


def main(root="./net_model_dataset"):
    for vnf in ("ran", "ovs", "upf"):
        d = os.path.join(root, vnf)
        os.makedirs(d, exist_ok=True)
        inp, out = _make_vnf(vnf)
        inp.to_pickle(os.path.join(d, "input_dataset.pkl"))
        out.to_pickle(os.path.join(d, "output_dataset.pkl"))
        print(f"  {vnf}: {len(inp)} rows -> {d}")
    print("synthetic dataset written under", root)


if __name__ == "__main__":
    main()
