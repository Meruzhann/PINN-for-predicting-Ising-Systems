import os
import numpy as np

OUT_DIR = "data_grid"

BETA_VALUES = np.linspace(0.3, 1.5, 5)
H_VALUES    = np.linspace(-2, 2, 5)
J_VALUES    = np.linspace(-1.5, 1.5, 4)

N_CHAIN = 256
TOTAL_TIME = 100.0
T_STEPS = 100
NUM_POINTS = 128
N_TRAJ_PER_COND = 100
MASTER_SEED = 20260503

def build_manifest():
    conds = []
    cond_idx = 0
    for beta in BETA_VALUES:
        for h in H_VALUES:
            for J in J_VALUES:
                conds.append({
                    "cond_idx": cond_idx,
                    "beta": float(beta),
                    "h": float(h),
                    "J": float(J),
                })
                cond_idx += 1
    return conds

def chunk_for(chunk_id, n_chunks):
    manifest = build_manifest()
    n_per_chunk = len(manifest) // n_chunks
    start = (chunk_id - 1) * n_per_chunk
    end = chunk_id * n_per_chunk if chunk_id < n_chunks else len(manifest)
    return manifest[start:end]

def run_one_trajectory(seed, beta, h, J,
                       num_points=NUM_POINTS, total_time=TOTAL_TIME,
                       sample_n=T_STEPS, N=N_CHAIN, dt=1):
    rng = np.random.RandomState(seed)
    T = np.arange(0, total_time, dt)
    all_points = np.zeros((T.shape[0], N), dtype=np.uint16)
    sample_rate = total_time / sample_n
    for p in range(num_points):

        seq_rng = np.random.RandomState(seed + 17 * p)
        seq = seq_rng.randint(0, 2, size=N, dtype=np.uint8)
        all_points[0] += seq
        t = 0.0
        ts = 0
        while t < total_time:
            idx = rng.randint(0, N)
            s_i = 2 * seq[idx] - 1
            s_nb = (2 * seq[(idx - 1) % N] - 1) + (2 * seq[(idx + 1) % N] - 1)
            mu_i = h + J * s_nb
            rate = 0.5 * (1 - s_i * np.tanh(beta * mu_i))
            U = rng.random()
            t -= np.log(1 - rng.random()) / N
            if rate > U:
                seq[idx] = 1 - seq[idx]
            if t > (ts + 1) * sample_rate:
                all_points[min(int(t / dt), T.shape[0] - 1)] += seq
                ts += 1
    return all_points

def generate_condition(cond, n_traj=N_TRAJ_PER_COND, n_jobs=1, verbose=True):
    cond_idx = cond["cond_idx"]
    seeds    = [MASTER_SEED + cond_idx * 100_000 + t for t in range(n_traj)]
    if n_jobs in (None, 0, 1):
        trajectories = [run_one_trajectory(s, cond["beta"], cond["h"], cond["J"])
                        for s in seeds]
    else:
        from joblib import Parallel, delayed
        trajectories = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(run_one_trajectory)(s, cond["beta"], cond["h"], cond["J"])
            for s in seeds)
    return np.stack(trajectories, axis=0)

def save_condition(cond, trajectories):
    os.makedirs(OUT_DIR, exist_ok=True)
    fname = os.path.join(OUT_DIR, f"cond_{cond['cond_idx']:03d}.npz")
    np.savez(fname,
             trajectories=trajectories,
             beta=cond["beta"], h=cond["h"], J=cond["J"],
             cond_idx=cond["cond_idx"],
             total_time=TOTAL_TIME,
             num_points=NUM_POINTS,
             N=N_CHAIN,
             master_seed=MASTER_SEED)
    return fname

def save_manifest():
    os.makedirs(OUT_DIR, exist_ok=True)
    m = build_manifest()
    arr = np.array([(c["cond_idx"], c["beta"], c["h"], c["J"]) for c in m],
                   dtype=[("cond_idx", "i4"), ("beta", "f8"), ("h", "f8"), ("J", "f8")])
    np.savez(os.path.join(OUT_DIR, "manifest.npz"), manifest=arr)

    csv_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(csv_path, "w") as f:
        f.write("cond_idx,beta,h,J\n")
        for c in m:
            f.write(f"{c['cond_idx']},{c['beta']:.6f},{c['h']:.6f},{c['J']:.6f}\n")
    return csv_path

def load_manifest():
    npz = os.path.join(OUT_DIR, "manifest.npz")
    if not os.path.exists(npz):
        return build_manifest()
    arr = np.load(npz, allow_pickle=False)["manifest"]
    return [{"cond_idx": int(r["cond_idx"]),
             "beta": float(r["beta"]), "h": float(r["h"]), "J": float(r["J"])}
            for r in arr]

def run_chunk(chunk_id, n_chunks=4, n_jobs=4):
    import time
    save_manifest()
    conds = chunk_for(chunk_id, n_chunks)
    print(f"Chunk {chunk_id}/{n_chunks}: {len(conds)} conditions, "
          f"{N_TRAJ_PER_COND} trajectories each → {len(conds) * N_TRAJ_PER_COND} total")
    print(f"Inner parallelism: n_jobs={n_jobs}")
    t_chunk = time.time()
    for c in conds:
        out = os.path.join(OUT_DIR, f"cond_{c['cond_idx']:03d}.npz")
        if os.path.exists(out):
            print(f"cond {c['cond_idx']:03d} (β={c['beta']:.3f} h={c['h']:.3f} J={c['J']:.3f})")
            continue
        t0 = time.time()
        traj = generate_condition(c, n_jobs=n_jobs)
        save_condition(c, traj)
        print(f"cond {c['cond_idx']:03d} (β={c['beta']:.3f} h={c['h']:.3f} J={c['J']:.3f})"
              f"→ {traj.shape} in {time.time() - t0:.1f}s")
    print(f"Chunk {chunk_id} complete in {(time.time() - t_chunk)/60:.1f} min")
