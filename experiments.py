"""
experiments.py - Chay toan bo thi nghiem, luu ket qua vao results/.

    python experiments.py           # day du: 5 seed moi thuat toan
    python experiments.py --quick   # thu nhanh: 2 seed, 1/10 so episode

Phan 1: moi map x 3 bien the (Q-learning Q0=0, Q-learning Q0=30, Monte Carlo) x nhieu seed.
Phan 2: doi reward / tham so moi truong (phat tre, do dai deadline, bo o tang tai).
"""
import argparse
import copy
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from agents import monte_carlo, q_learning, run_greedy
from delivery_env import MultiDeliveryEnv, load_config
from planner import value_iteration

RESULTS = "results"

# Ten bien the -> (nhan hien thi, ham hoc, tham so rieng, he so nhan so episode)
ALGOS = {
    "ql_q0": ("Q-learning (Q0 = 0)", q_learning, {"q_init": 0.0}, 1),
    "ql_opt": ("Q-learning (Q0 = 30, lac quan)", q_learning, {"q_init": 30.0}, 1),
    "mc": ("Monte Carlo first-visit", monte_carlo, {}, 1),
    # MC "keo dai": 4 lan so episode, eps giam cham hon (chi chay tren map nho: train.mc_long = true)
    "mc_long": ("Monte Carlo, 4x episode, eps giam cham", monte_carlo, {"eps_end": 0.1, "decay_frac": 0.9}, 4),
}

# Phan 2: (ten, file cau hinh goc, ham sua cau hinh, mo ta)
def _late0(c): c["rewards"]["late_penalty"] = 0
def _long_deadline(c): c["parcels"][2]["deadline"] = 35
def _no_booster(c): c["grid"] = [row.replace("B", ".") for row in c["grid"]]
def _same(c): pass

VARIANTS = [
    ("S3_phat_tre_30", "configs/s3_deadline.json", _same, "Deadline 18 buoc, phat tre 30 (goc)"),
    ("S3_phat_tre_0", "configs/s3_deadline.json", _late0, "Deadline 18 buoc, phat tre 0"),
    ("S3_deadline_35", "configs/s3_deadline.json", _long_deadline, "Deadline 35 buoc, phat tre 30"),
    ("S4_co_tang_tai", "configs/s4_booster.json", _same, "Co o tang tai (goc)"),
    ("S4_khong_tang_tai", "configs/s4_booster.json", _no_booster, "Bo o tang tai"),
]


def train_one(job):
    """Mot lan huan luyen (1 cau hinh, 1 thuat toan, 1 seed). Chay trong process con."""
    cfg, key, seed, scale = job
    tr = cfg["train"]
    _, fn, extra, mult = ALGOS[key]
    env = MultiDeliveryEnv(cfg, tr["start_mode"])
    ev = MultiDeliveryEnv(cfg, "depot")
    kw = dict(n_episodes=max(200, int(tr["episodes"] * scale * mult)), gamma=tr["gamma"],
              eps_start=tr["eps_start"], eps_end=tr["eps_end"], decay_frac=tr["decay_frac"],
              seed=seed, eval_every=tr["eval_every"] * mult)
    kw.update(extra)
    if fn is q_learning:
        kw["alpha"] = tr["alpha_ql"]
    t0 = time.time()
    Q, hist = fn(env, ev, **kw)
    final = run_greedy(ev, Q)
    return {"name": cfg["name"], "algo": key, "seed": seed, "time": time.time() - t0,
            "hist": hist, "Q": Q if seed == 0 else None,
            "final": {k: final[k] for k in ("return", "success", "steps", "events")}}


def route_summary(res):
    """Thu tu giao kien + so buoc + co tre deadline khong."""
    order = [e.split()[2] for _, e in res["events"] if e.startswith("giao kien")]
    late = any("tre deadline" in e for _, e in res["events"])
    return {"return": res["return"], "steps": res["steps"], "success": res["success"],
            "order": "-".join(order), "late": late}


def optimal_info(cfg):
    ev = MultiDeliveryEnv(cfg, "depot")
    Qv, n_reach = value_iteration(ev, gamma=cfg["train"]["gamma"])
    return ev, Qv, n_reach, route_summary(run_greedy(ev, Qv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    seeds = [0, 1] if args.quick else [0, 1, 2, 3, 4]
    scale = 0.1 if args.quick else 1.0
    os.makedirs(RESULTS, exist_ok=True)

    configs = [load_config(p) for p in sorted(glob.glob("configs/*.json"))]
    summary = {"seeds": seeds, "settings": {}, "variants": {}}

    # ---------- Kich thuoc state space + policy toi uu (Value Iteration) ----------
    for cfg in configs:
        ev, Qv, n_reach, opt = optimal_info(cfg)
        summary["settings"][cfg["name"]] = {
            "title": cfg["title"], "grid": f"{ev.H}x{ev.W}", "n_parcels": ev.N,
            "capacity": ev.capacity, "n_pos": ev.n_pos, "n_codes": ev.n_codes,
            "n_tau": ev.n_tau, "n_boost": ev.n_boost, "n_states": ev.n_states,
            "n_reachable": n_reach, "n_actions": int(ev.action_space.n), "optimal": opt,
            "episodes": max(200, int(cfg["train"]["episodes"] * scale)), "algos": {}}
        print(f"{cfg['name']}: |S| = {ev.n_states}, dat toi = {n_reach}, toi uu = {opt}", flush=True)

    # ---------- Phan 1: huan luyen song song ----------
    jobs = [(cfg, key, s, scale) for cfg in configs for key in ALGOS for s in seeds
            if key != "mc_long" or cfg["train"].get("mc_long")]
    jobs.sort(key=lambda j: -j[0]["train"]["episodes"] * ALGOS[j[1]][3])  # job dai chay truoc
    t0 = time.time()
    with Pool(args.workers) as pool:
        outs = []
        for i, out in enumerate(pool.imap_unordered(train_one, jobs), 1):
            outs.append(out)
            print(f"  [{i}/{len(jobs)}] {out['name']} {out['algo']} seed={out['seed']} "
                  f"return={out['final']['return']:.0f} ({out['time']:.0f}s)", flush=True)
    print(f"Phan 1 xong sau {time.time() - t0:.0f}s", flush=True)

    for cfg in configs:
        name = cfg["name"]
        st = summary["settings"][name]
        arrays = {}
        for key, (label, _, _, _) in ALGOS.items():
            runs = sorted([o for o in outs if o["name"] == name and o["algo"] == key], key=lambda o: o["seed"])
            if not runs:
                continue
            ev_ret = np.array([o["hist"]["eval_return"] for o in runs])
            ev_suc = np.array([o["hist"]["eval_success"] for o in runs], dtype=float)
            arrays[f"{key}__eval_ep"] = np.array(runs[0]["hist"]["eval_ep"])
            arrays[f"{key}__eval_return"] = ev_ret
            arrays[f"{key}__eval_success"] = ev_suc
            arrays[f"{key}__train_len"] = np.array([o["hist"]["train_len"] for o in runs])
            arrays[f"{key}__train_return"] = np.array([o["hist"]["train_return"] for o in runs])
            np.save(os.path.join(RESULTS, f"Q_{name}_{key}.npy"), runs[0]["Q"])

            opt_ret = st["optimal"]["return"]
            first_opt = []
            for row in ev_ret:
                hit = np.flatnonzero(row >= opt_ret - 1e-9)
                first_opt.append(int(arrays[f"{key}__eval_ep"][hit[0]]) if len(hit) else None)
            finals = [o["final"] for o in runs]
            st["algos"][key] = {
                "label": label,
                "final_return_mean": float(np.mean([f["return"] for f in finals])),
                "final_return_std": float(np.std([f["return"] for f in finals])),
                "success_rate": float(np.mean([f["success"] for f in finals])),
                "optimal_rate": float(np.mean([f["return"] >= opt_ret - 1e-9 for f in finals])),
                "first_optimal_episode": first_opt,
                "final_steps": [f["steps"] for f in finals],
                "final_routes": [route_summary(f) for f in finals],
                "time_mean_s": float(np.mean([o["time"] for o in runs])),
            }
        np.savez_compressed(os.path.join(RESULTS, f"{name}.npz"), **arrays)

    # ---------- Phan 2: doi reward / tham so moi truong ----------
    base = {os.path.normpath(p): load_config(p) for p in glob.glob("configs/*.json")}
    var_jobs, var_meta = [], {}
    for vname, path, modify, desc in VARIANTS:
        cfg = copy.deepcopy(base[os.path.normpath(path)])
        modify(cfg)
        cfg["name"] = vname
        _, _, n_reach, opt = optimal_info(cfg)
        var_meta[vname] = {"desc": desc, "base": path, "optimal": opt, "n_reachable": n_reach}
        algo = cfg["train"]["best_ql"]
        cfg["train"]["episodes"] *= 2  # bien the: gap doi so episode (deadline dai -> state space lon hon)
        var_jobs += [(cfg, algo, s, scale) for s in seeds]
        with open(os.path.join(RESULTS, f"config_{vname}.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
    t0 = time.time()
    with Pool(args.workers) as pool:
        var_outs = pool.map(train_one, var_jobs)
    print(f"Phan 2 xong sau {time.time() - t0:.0f}s", flush=True)
    for vname, meta in var_meta.items():
        runs = sorted([o for o in var_outs if o["name"] == vname], key=lambda o: o["seed"])
        meta["algo"] = runs[0]["algo"]
        meta["learned_routes"] = [route_summary(o["final"]) for o in runs]
        np.save(os.path.join(RESULTS, f"Q_variant_{vname}.npy"), runs[0]["Q"])
        print(f"{vname}: toi uu {meta['optimal']} | hoc duoc {meta['learned_routes'][0]}", flush=True)
    summary["variants"] = var_meta

    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("Da luu results/summary.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
