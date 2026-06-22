#!/usr/bin/env python
"""
End-to-end smoke / integration test for StreamCore on the synthetic dataset.
Exercises: data loading -> per-VNF training -> slice prediction -> resource
allocation, plus the two previously-crashing edge paths. Exits non-zero on
failure so it can gate CI.

Run:
    python tools/generate_synthetic_dataset.py
    python tools/smoke_test.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from data_generator import DataGenerator
from vnf_model import VNF_Model
from slice_model import SliceModel
import resource_allocation as RA

EPOCHS = int(os.environ.get("SMOKE_EPOCHS", "400"))


def main():
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs("./data", exist_ok=True)
    gens, models = {}, {}
    for v, t in [("ran", "RAN"), ("ovs", "OvS"), ("upf", "UPF")]:
        gens[v] = DataGenerator(f"./net_model_dataset/{v}/input_dataset.pkl",
                                f"./net_model_dataset/{v}/output_dataset.pkl",
                                vnf_type=t, norm_type="minmax")
    arch = {"ran": [64, 32, 16], "ovs": [64, 32, 16], "upf": [32, 16]}
    for v in ("ran", "ovs", "upf"):
        m = VNF_Model(vnf_typ=v, n_inputs=6, n_hidden=arch[v], n_outputs=5)
        m.fit(gens[v], num_epochs=EPOCHS, batch_size=256, save_model=False, save_loss=False)
        models[v] = m

    slice_model = SliceModel([models["upf"], models["ovs"], models["ran"]],
                             [gens["upf"], gens["ovs"], gens["ran"]])

    ok = True
    tp = slice_model.predict_throughput(res=[200, 20, 1000], input_throughput=35,
                                        differentiable=0, res_normalized=False)
    print(f"[1] slice throughput = {tp:.3f} Mbps")
    if tp < 0:
        print("    FAIL: negative throughput"); ok = False

    # allocator happy path
    ra, qos, ub, lb, t = RA.allocate_resources(slice_model, input_throughput=20,
                                               qos_threshold=15, verbose=0)
    print(f"[2] allocate_resources -> {None if ra is None else np.round(ra.flatten(),3)}, "
          f"qos={None if qos is None else round(qos,2)}, {t:.2f}s")

    # edge path: qos target > input (used to NameError on start_time)
    r = RA.allocate_resources(slice_model, input_throughput=10, qos_threshold=20, verbose=0)
    print(f"[3] qos>input edge path returned cleanly: {r[0] is None}")

    # edge path: infeasible grid (used to crash get_initial_solution on None)
    try:
        RA.allocate_resources(slice_model, input_throughput=59, qos_threshold=58.9, verbose=0)
        print("[4] infeasible-grid path did not crash")
    except AttributeError as e:
        print(f"[4] FAIL: {e}"); ok = False

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
