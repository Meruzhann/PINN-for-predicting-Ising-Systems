import os
import numpy as np
import torch
import torch.nn as nn

DATA_DIR = 'data_grid'
DATA_FILE = 'cond_046.npz'
TOTAL_TIME = 100.0
T_KEEP = 99
HIDDEN = 64
DEPTH = 4
LR = 1e-3
EPOCHS_ADAM = 4000
LBFGS_STEPS = 300
N_COLLOC = 200
SEED = 0

def load_condition(data_path=None, return_per_run=False):
    path = data_path or os.path.join(DATA_DIR, DATA_FILE)
    d = np.load(path)
    tr = d['trajectories'][:, :T_KEEP, :]
    p = tr.astype(np.float32) / int(d['num_points'])

    spins_all = (2.0 * p - 1.0).astype(np.float32)
    m_per_run = (2.0 * p.mean(axis=2) - 1.0).astype(np.float32)
    m_bar = m_per_run.mean(axis=0)
    nn_prod = spins_all * np.roll(spins_all, -1, axis=2)
    nn2_prod = spins_all * np.roll(spins_all, -2, axis=2)
    eps_bar = nn_prod.mean(axis=2).mean(axis=0).astype(np.float32)
    eps2_bar = nn2_prod.mean(axis=2).mean(axis=0).astype(np.float32)

    t_grid = np.linspace(0.0, TOTAL_TIME, T_KEEP + 1,
                         endpoint=False)[:T_KEEP].astype(np.float32)

    out = dict(
        t_grid = t_grid,
        m_bar = m_bar,
        eps_bar = eps_bar,
        eps2_bar = eps2_bar,
        beta_true = float(d['beta']),
        h_true = float(d['h']),
        J_true = float(d['J']),
    )
    if return_per_run:
        out['m_per_run'] = m_per_run
    return out

def glauber_constants(beta, h, J):
    f_p = np.tanh(beta * (h + 2.0 * J))
    f_0 = np.tanh(beta * h)
    f_m = np.tanh(beta * (h - 2.0 * J))
    a0 = 0.5 * (f_p + 2.0 * f_0 + f_m)
    a1 = 0.5 * (f_p - f_m)
    a2 = 0.5 * (f_p - 2.0 * f_0 + f_m)
    return float(a0), float(a1), float(a2)

def glauber_constants_torch(beta, h, J):
    f_p = torch.tanh(beta * (h + 2.0 * J))
    f_0 = torch.tanh(beta * h)
    f_m = torch.tanh(beta * (h - 2.0 * J))
    a0 = 0.5 * (f_p + 2.0 * f_0 + f_m)
    a1 = 0.5 * (f_p - f_m)
    a2 = 0.5 * (f_p - 2.0 * f_0 + f_m)
    return a0, a1, a2

def make_mlp(hidden=HIDDEN, depth=DEPTH, n_in=1, n_out=1):
    layers = [nn.Linear(n_in, hidden), nn.Softplus()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Softplus()]
    layers += [nn.Linear(hidden, n_out)]
    return nn.Sequential(*layers)

def scalar_init(value, scale):
    v = max(min(float(value) / scale, 0.99), -0.99)
    return torch.tensor(np.arctanh(v), dtype=torch.float32)

def time_derivative(y, t):
    return torch.autograd.grad(
        y, t, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adam_then_lbfgs(closure_loss, params, n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS,
                    lr=LR, lbfgs_max_iter=25):
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for _ in range(n_adam):
        L = closure_loss()
        opt.step()
        history.append(float(L))

    opt_l = torch.optim.LBFGS(
        params, lr=1.0, max_iter=lbfgs_max_iter, history_size=100,
        line_search_fn='strong_wolfe',
        tolerance_grad=1e-12, tolerance_change=1e-15,
    )
    last = [None]
    def cl():
        L = closure_loss()
        last[0] = L
        return L

    for _ in range(n_lbfgs):
        snapshot = [p.detach().clone() for p in params]
        try:
            opt_l.step(cl)
        except RuntimeError:

            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)
            break
        if last[0] is None or not torch.isfinite(last[0]):
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)
            break
        history.append(float(last[0]))
    return history

def mse(pred, truth):
    return float(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2))
