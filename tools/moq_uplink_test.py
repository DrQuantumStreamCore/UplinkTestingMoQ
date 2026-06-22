#!/usr/bin/env python
"""
MoQ (Media over QUIC) uplink transmission test for StreamCore.

WHAT THIS IS
------------
A runnable *simulation* that drives StreamCore with a MoQ-like UPLINK media
traffic model and exercises the new uplink slice path + resource allocator. It
is NOT a live MoQ relay / QUIC stack -- it models the traffic characteristics
that matter for resource allocation:

  * Object/group cadence: MoQ publishes media as a stream of objects grouped
    into groups (e.g. one group per GoP). Each group starts with a large
    key-object burst followed by smaller delta objects.
  * Uplink-heavy: the contributor (camera / "see-what-I-see") pushes media UP
    to the relay, so the offered load is on the uplink -- the direction that is
    now the congestion point.
  * Burstiness: per-group bursts make the offered rate spiky, which is exactly
    what stresses RAN uplink scheduling.

WHAT IT CHECKS
--------------
  1. The slice can be evaluated in the 'uplink' direction (RAN -> OvS -> UPF).
  2. Downlink vs uplink composition give different throughput for the same load
     (sanity that direction actually changes the computation).
  3. The allocator can dimension resources to sustain a MoQ uplink bitrate
     target, optimising in the uplink direction.

Run:
    python tools/generate_synthetic_dataset.py    # if not already present
    python tools/moq_uplink_test.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from data_generator import DataGenerator
from vnf_model import VNF_Model
from slice_model import SliceModel
import resource_allocation as RA

RNG = np.random.default_rng(11)


# --- MoQ uplink traffic model -------------------------------------------------
def moq_uplink_trace(target_mbps=12.0, fps=30, gop=30, seconds=4, key_ratio=6.0):
    """Generate a per-frame MoQ uplink offered-load trace (Mbps).

    target_mbps : average uplink media bitrate the contributor pushes.
    fps         : frames per second (objects per second).
    gop         : group length in frames (key object every `gop` frames).
    key_ratio   : size multiplier of a key object vs an average delta object.
    Returns an array of instantaneous offered throughput per frame (Mbps).
    """
    n = fps * seconds
    # base per-object size so the long-run average equals target_mbps
    avg_bits_per_frame = target_mbps * 1e6 / fps
    sizes = np.empty(n)
    for i in range(n):
        is_key = (i % gop == 0)
        sizes[i] = key_ratio if is_key else 1.0
    sizes *= avg_bits_per_frame / sizes.mean()           # renormalise to target
    sizes *= RNG.uniform(0.85, 1.15, n)                  # encoder jitter
    inst_mbps = sizes * fps / 1e6                        # instantaneous rate
    return inst_mbps


def train_models(epochs=500):
    gens, models = {}, {}
    for v, t in [("ran", "RAN"), ("ovs", "OvS"), ("upf", "UPF")]:
        gens[v] = DataGenerator(f"./net_model_dataset/{v}/input_dataset.pkl",
                                f"./net_model_dataset/{v}/output_dataset.pkl",
                                vnf_type=t, norm_type="minmax")
    arch = {"ran": [64, 32, 16], "ovs": [64, 32, 16], "upf": [32, 16]}
    for v in ("ran", "ovs", "upf"):
        m = VNF_Model(vnf_typ=v, n_inputs=6, n_hidden=arch[v], n_outputs=5)
        m.fit(gens[v], num_epochs=epochs, batch_size=256, save_model=False, save_loss=False)
        models[v] = m
    # stored order = downlink (UPF -> OvS -> RAN); uplink reverses it
    slice_model = SliceModel([models["upf"], models["ovs"], models["ran"]],
                             [gens["upf"], gens["ovs"], gens["ran"]])
    return slice_model


def main():
    if not os.path.exists("./net_model_dataset/ran/input_dataset.pkl"):
        print("No dataset found -- run tools/generate_synthetic_dataset.py first.")
        sys.exit(2)
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs("./data", exist_ok=True)
    print("Training per-VNF models on synthetic data ...")
    slice_model = train_models()

    target = 12.0
    trace = moq_uplink_trace(target_mbps=target)
    res = [200, 20, 1500]   # UPF cpu, OvS bw, RAN cpu (stored order)

    print("\n=== MoQ uplink trace ===")
    print(f"frames={len(trace)}  mean={trace.mean():.2f} Mbps  "
          f"peak={trace.max():.2f} Mbps  burst_ratio={trace.max()/trace.mean():.2f}x")

    # [1+2] downlink vs uplink composition on the peak group-burst load
    peak = float(trace.max())
    dl = slice_model.predict_throughput(res, peak, differentiable=0, res_normalized=False, direction="downlink")
    ul = slice_model.predict_throughput(res, peak, differentiable=0, res_normalized=False, direction="uplink")
    print("\n=== Direction sanity (peak burst offered load) ===")
    print(f"offered peak      : {peak:.2f} Mbps")
    print(f"downlink sustained: {dl:.2f} Mbps")
    print(f"uplink   sustained: {ul:.2f} Mbps   (this is the congested direction)")

    # [3] dimension resources to sustain the MoQ uplink target, in uplink order
    print("\n=== Uplink resource allocation for MoQ target ===")
    qos_target = target * 0.95   # sustain 95% of mean uplink bitrate
    ra, qos, ub, lb, t = RA.allocate_resources(
        slice_model, input_throughput=target, qos_threshold=qos_target,
        verbose=0, direction="uplink")
    if ra is None:
        print("allocator returned no solution for this target")
    else:
        gens = slice_model.data_gens
        denorm = [float(gens[i].denormalize(ra.flatten()[i], feature_type="input", feature="res"))
                  for i in range(3)]
        names = ["UPF cpu(mc)", "OvS bw(Mbps)", "RAN cpu(mc)"]
        alloc = {n: round(v, 1) for n, v in zip(names, denorm)}
        print(f"target uplink bitrate : {target:.1f} Mbps (sustain >= {qos_target:.1f})")
        print(f"allocation (uplink)   : {alloc}")
        print(f"achieved QoS          : {qos:.2f} Mbps   in {t:.2f}s")

    print("\nNOTE: synthetic data + traffic model -> this validates the uplink")
    print("pipeline mechanics, not real MoQ-over-QUIC numbers. Swap in testbed")
    print("captures (pcap -> dataset) to get production figures.")


if __name__ == "__main__":
    main()
