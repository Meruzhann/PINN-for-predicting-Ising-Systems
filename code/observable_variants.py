import time
import numpy as np
import torch
import torch.nn as nn

from shared import (
    HIDDEN, DEPTH, LR, EPOCHS_ADAM, LBFGS_STEPS, N_COLLOC, TOTAL_TIME, SEED,
    make_mlp, scalar_init, time_derivative, count_params,
    adam_then_lbfgs, mse,
)
from variants import BoundedScalars, PINN_m_Core

def start_from_m(data, device='cpu'):
    t_grid = data['t_grid']
    m_bar = data['m_bar']
    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t = torch.tensor(m_bar,  dtype=torch.float32, device=device).view(-1, 1)

    torch.manual_seed(SEED)
    model = PINN_m_Core().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    def closure():
        opt.zero_grad()
        L_d = torch.mean((model(t_t) - m_t) ** 2)
        tc  = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        m_c = model(tc)
        m_dot = time_derivative(m_c, tc)
        res   = m_dot + (1.0 - model.a1) * m_c - 0.5 * model.a0 - 0.5 * model.a2 * m_c ** 2
        L_p   = torch.mean(res ** 2)
        L = L_d + L_p
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(closure, list(model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)
    train_time = time.time() - t0

    with torch.no_grad():
        m_hat = model(t_t).cpu().numpy().squeeze()
        a0, a1, a2 = model.a0.item(), model.a1.item(), model.a2.item()
    eps_hat = m_hat ** 2
    t_grad = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    t_grad.requires_grad_(True)
    m_g    = model(t_grad)
    m_dot  = time_derivative(m_g, t_grad).detach().cpu().numpy().squeeze()
    eps2_hat = (2.0 / a2) * (m_dot + (1.0 - a1) * m_hat - 0.5 * a0) if abs(a2) > 1e-6 else m_hat ** 2

    return {
        'name': 'Start from m',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics_native': True,
    }

class PINN_eps_Core(nn.Module):
    def __init__(self, total_time=TOTAL_TIME):
        super().__init__()
        self.total_time = float(total_time)
        self.net = make_mlp(hidden=HIDDEN, depth=DEPTH, n_in=1, n_out=1)
        self.scalars = BoundedScalars()

    @property
    def a0(self): return self.scalars.a0
    @property
    def a1(self): return self.scalars.a1
    @property
    def a2(self): return self.scalars.a2

    def forward(self, t):
        return torch.tanh(self.net(t / self.total_time))

def start_from_eps(data, device='cpu'):
    t_grid    = data['t_grid']
    m_bar     = data['m_bar']
    eps_bar   = data['eps_bar']
    eps2_bar  = data['eps2_bar']

    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    e_t = torch.tensor(eps_bar, dtype=torch.float32, device=device).view(-1, 1)

    def interp_obs(values, t_eval):
        t_np = t_eval.detach().cpu().numpy().squeeze()
        v    = np.interp(t_np, t_grid, values)
        return torch.tensor(v, dtype=torch.float32, device=device).view(-1, 1)

    torch.manual_seed(SEED)
    model = PINN_eps_Core().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def closure():
        opt.zero_grad()
        L_d = torch.mean((model(t_t) - e_t) ** 2)
        tc  = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        e_c = model(tc)
        e_dot = time_derivative(e_c, tc)
        m_c   = interp_obs(m_bar,    tc)
        e2_c  = interp_obs(eps2_bar, tc)
        res   = e_dot + 2.0 * e_c - model.a1 - (model.a0 + model.a2) * m_c - model.a1 * e2_c
        L_p   = torch.mean(res ** 2)
        L = L_d + L_p
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(closure, list(model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)
    train_time = time.time() - t0

    with torch.no_grad():
        eps_hat = model(t_t).cpu().numpy().squeeze()
        a0, a1, a2 = model.a0.item(), model.a1.item(), model.a2.item()

    t_grad = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    t_grad.requires_grad_(True)
    e_g    = model(t_grad)
    e_dot  = time_derivative(e_g, t_grad).detach().cpu().numpy().squeeze()
    denom  = (a0 + a2)
    if abs(denom) > 1e-6:
        m_hat = (e_dot + 2.0 * eps_hat - a1 - a1 * eps2_bar) / denom
    else:
        m_hat = np.sqrt(np.clip(eps_hat, 0.0, None))

    eps2_hat = eps2_bar.copy()

    return {
        'name': 'Start from ε',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics_native': True,
    }

class PINN_eps2_Core(nn.Module):
    def __init__(self, total_time=TOTAL_TIME):
        super().__init__()
        self.total_time = float(total_time)
        self.net = make_mlp(hidden=HIDDEN, depth=DEPTH, n_in=1, n_out=1)

    def forward(self, t):
        return torch.tanh(self.net(t / self.total_time))

def start_from_eps2(data, device='cpu'):
    t_grid    = data['t_grid']
    m_bar     = data['m_bar']
    eps2_bar  = data['eps2_bar']

    t_t  = torch.tensor(t_grid,  dtype=torch.float32, device=device).view(-1, 1)
    e2_t = torch.tensor(eps2_bar, dtype=torch.float32, device=device).view(-1, 1)

    torch.manual_seed(SEED)
    model = PINN_eps2_Core().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def closure():
        opt.zero_grad()
        L = torch.mean((model(t_t) - e2_t) ** 2)
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(closure, list(model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)

    with torch.no_grad():
        eps2_hat = model(t_t).cpu().numpy().squeeze()

    m_dot = np.gradient(m_bar, t_grid)
    y = m_dot + m_bar
    X = np.stack([m_bar, np.ones_like(m_bar), 0.5 * eps2_hat], axis=1)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a1 = float(coefs[0])
    a0 = float(2.0 * coefs[1])
    a2 = float(2.0 * coefs[2])
    train_time = time.time() - t0

    m_hat   = m_bar.copy()
    eps_hat = m_bar ** 2

    return {
        'name': 'Start from ε₂',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics_native': False,
    }
