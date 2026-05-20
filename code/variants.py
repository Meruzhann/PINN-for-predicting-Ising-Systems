import time
import numpy as np
import torch
import torch.nn as nn

from shared import (
    HIDDEN, DEPTH, LR, EPOCHS_ADAM, LBFGS_STEPS, N_COLLOC, TOTAL_TIME, T_KEEP, SEED,
    make_mlp, scalar_init, time_derivative, count_params,
    adam_then_lbfgs, glauber_constants_torch, mse, load_condition,
)

class BoundedScalars(nn.Module):
    def __init__(self, seed_offset=0):
        super().__init__()
        rng = np.random.default_rng(SEED + seed_offset)
        self.a0_raw = nn.Parameter(scalar_init(float(rng.uniform(-0.5, 0.5)), 2.0))
        self.a1_raw = nn.Parameter(scalar_init(float(rng.uniform(-0.3, 0.3)), 1.0))
        self.a2_raw = nn.Parameter(scalar_init(float(rng.uniform(-0.5, 0.5)), 2.0))

    @property
    def a0(self): return 2.0 * torch.tanh(self.a0_raw)
    @property
    def a1(self): return torch.tanh(self.a1_raw)
    @property
    def a2(self): return 2.0 * torch.tanh(self.a2_raw)

class PINN_m_Core(nn.Module):
    OUTPUT_SCALE = 1.05

    def __init__(self, total_time=TOTAL_TIME, hidden=HIDDEN, depth=DEPTH, seed_offset=0):
        super().__init__()
        self.total_time = float(total_time)
        self.net = make_mlp(hidden=hidden, depth=depth, n_in=1, n_out=1)
        self.scalars = BoundedScalars(seed_offset=seed_offset)

    @property
    def a0(self): return self.scalars.a0
    @property
    def a1(self): return self.scalars.a1
    @property
    def a2(self): return self.scalars.a2

    def forward(self, t):
        return self.OUTPUT_SCALE * torch.tanh(self.net(t / self.total_time))

def _make_train_mask(t_grid, t_train_max, device):
    if t_train_max is None:
        mask = np.ones_like(t_grid, dtype=bool)
    else:
        mask = t_grid < float(t_train_max)
        mask[-1] = True
    idx = np.where(mask)[0]
    return torch.tensor(idx, dtype=torch.long, device=device), mask

def three_separate_pinns(data, device='cpu', t_train_max=None):
    t_grid = data['t_grid']
    m_bar = data['m_bar']
    eps_bar = data['eps_bar']
    eps2_bar = data['eps2_bar']

    t_t   = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t   = torch.tensor(m_bar, dtype=torch.float32, device=device).view(-1, 1)
    e_t   = torch.tensor(eps_bar, dtype=torch.float32, device=device).view(-1, 1)
    e2_t  = torch.tensor(eps2_bar, dtype=torch.float32, device=device).view(-1, 1)
    train_idx, train_mask = _make_train_mask(t_grid, t_train_max, device)

    torch.manual_seed(SEED)
    m_model = PINN_m_Core(seed_offset=0).to(device)
    opt_m = torch.optim.Adam(m_model.parameters(), lr=LR)
    def m_closure():
        opt_m.zero_grad()
        L_d = torch.mean((m_model(t_t)[train_idx] - m_t[train_idx]) ** 2)
        tc= (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        m_c = m_model(tc)
        m_dot = time_derivative(m_c, tc)
        res = m_dot + (1.0 - m_model.a1) * m_c - 0.5 * m_model.a0 - 0.5 * m_model.a2 * m_c ** 2
        L = L_d + torch.mean(res ** 2)
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(m_closure, list(m_model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)

    torch.manual_seed(SEED + 1)
    e_model = PINN_m_Core(seed_offset=1).to(device)
    opt_e = torch.optim.Adam(e_model.parameters(), lr=LR)
    def interp_obs(values, t_eval):
        t_np = t_eval.detach().cpu().numpy().squeeze()
        v = np.interp(t_np, t_grid, values)
        return torch.tensor(v, dtype=torch.float32, device=device).view(-1, 1)
    def e_closure():
        opt_e.zero_grad()
        L_d = torch.mean((e_model(t_t)[train_idx] - e_t[train_idx]) ** 2)
        tc = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        e_c = e_model(tc)
        e_dot = time_derivative(e_c, tc)
        m_c = interp_obs(m_bar, tc)
        e2_c = interp_obs(eps2_bar, tc)
        res = e_dot + 2.0 * e_c - e_model.a1 - (e_model.a0 + e_model.a2) * m_c - e_model.a1 * e2_c
        L = L_d + torch.mean(res ** 2)
        L.backward()
        return L
    adam_then_lbfgs(e_closure, list(e_model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)

    torch.manual_seed(SEED + 2)
    e2_model = PINN_m_Core(seed_offset=2).to(device)
    opt_e2 = torch.optim.Adam(e2_model.parameters(), lr=LR)
    def e2_closure():
        opt_e2.zero_grad()
        L = torch.mean((e2_model(t_t)[train_idx] - e2_t[train_idx]) ** 2)
        L.backward()
        return L
    for _ in range(EPOCHS_ADAM):
        e2_closure()
        opt_e2.step()
    train_time = time.time() - t0

    with torch.no_grad():
        m_hat  = m_model(t_t).cpu().numpy().squeeze()
        e_hat  = e_model(t_t).cpu().numpy().squeeze()
        e2_hat = e2_model(t_t).cpu().numpy().squeeze()
        a0 = m_model.a0.item()
        a1 = m_model.a1.item()
        a2 = m_model.a2.item()
    n_params = (count_params(m_model) + count_params(e_model) + count_params(e2_model))

    return {
        'name': 'Three separate PINNs',
        'm_hat': m_hat, 'eps_hat': e_hat, 'eps2_hat': e2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': n_params, 'time_sec': train_time,
        'physics': True, 'full_supervised': True,
    }

class InversePINN(nn.Module):
    def __init__(self, total_time=TOTAL_TIME):
        super().__init__()
        self.total_time = float(total_time)
        self.net = make_mlp(hidden=HIDDEN, depth=DEPTH, n_in=1, n_out=1)
        rng = np.random.default_rng(SEED)
        self.beta_raw = nn.Parameter(torch.tensor(float(rng.uniform(0.3, 1.2))))
        self.h_raw = nn.Parameter(torch.tensor(float(rng.uniform(-0.3, 0.3))))
        self.J_raw = nn.Parameter(torch.tensor(float(rng.uniform(0.3, 1.2))))

    def coefficients(self):
        return glauber_constants_torch(self.beta_raw, self.h_raw, self.J_raw)

    def forward(self, t):
        return torch.tanh(self.net(t / self.total_time))

def inverse_pinn(data, device='cpu', t_train_max=None):
    t_grid = data['t_grid']
    m_bar = data['m_bar']
    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t = torch.tensor(m_bar,  dtype=torch.float32, device=device).view(-1, 1)
    train_idx, _ = _make_train_mask(t_grid, t_train_max, device)

    torch.manual_seed(SEED)
    model = InversePINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    def closure():
        opt.zero_grad()
        a0, a1, a2 = model.coefficients()
        L_d = torch.mean((model(t_t)[train_idx] - m_t[train_idx]) ** 2)
        tc = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        m_c = model(tc)
        m_dot = time_derivative(m_c, tc)
        res = m_dot + (1.0 - a1) * m_c - 0.5 * a0 - 0.5 * a2 * m_c ** 2
        L = L_d + torch.mean(res ** 2)
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(closure, list(model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)
    train_time = time.time() - t0

    with torch.no_grad():
        m_hat = model(t_t).cpu().numpy().squeeze()
        a0, a1, a2 = (float(x) for x in model.coefficients())

    eps_hat = m_hat ** 2
    t_grad = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    t_grad.requires_grad_(True)
    m_g = model(t_grad)
    m_dot = time_derivative(m_g, t_grad).detach().cpu().numpy().squeeze()
    eps2_hat = (2.0 / a2) * (m_dot + (1.0 - a1) * m_hat - 0.5 * a0) if abs(a2) > 1e-6 else m_hat ** 2

    return {
        'name': 'Inverse PINN (β, h, J learnable)',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'beta_recovered': float(model.beta_raw),
        'h_recovered': float(model.h_raw),
        'J_recovered': float(model.J_raw),
        'n_params': count_params(model), 'time_sec': train_time,
        'physics': True, 'full_supervised': False,
    }

def pinn_analytical(data, device='cpu', t_train_max=None):
    t_grid = data['t_grid']
    m_bar = data['m_bar']
    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t = torch.tensor(m_bar, dtype=torch.float32, device=device).view(-1, 1)
    train_idx, _ = _make_train_mask(t_grid, t_train_max, device)

    DATA_WEIGHT = 10.0

    def build_and_train(use_lbfgs=True):
        torch.manual_seed(SEED)
        model = PINN_m_Core().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        def closure():
            opt.zero_grad()
            L_d = torch.mean((model(t_t)[train_idx] - m_t[train_idx]) ** 2)
            tc = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
            m_c = model(tc)
            m_dot = time_derivative(m_c, tc)
            res = m_dot + (1.0 - model.a1) * m_c - 0.5 * model.a0 - 0.5 * model.a2 * m_c ** 2
            L = DATA_WEIGHT * L_d + torch.mean(res ** 2)
            L.backward()
            return L
        adam_then_lbfgs(closure, list(model.parameters()),
                        n_adam=EPOCHS_ADAM,
                        n_lbfgs=(LBFGS_STEPS if use_lbfgs else 0))
        return model

    t0 = time.time()
    model = build_and_train(use_lbfgs=True)

    with torch.no_grad():
        if not torch.isfinite(model(t_t)).all():
            print('[pinn_analytical] NaN detected after L-BFGS; '
                  'retrying with Adam-only')
            model = build_and_train(use_lbfgs=False)
    train_time = time.time() - t0

    with torch.no_grad():
        m_hat = model(t_t).cpu().numpy().squeeze()
        a0, a1, a2 = model.a0.item(), model.a1.item(), model.a2.item()

    eps_hat = m_hat ** 2
    t_grad = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    t_grad.requires_grad_(True)
    m_g = model(t_grad)
    m_dot = time_derivative(m_g, t_grad).detach().cpu().numpy().squeeze()
    eps2_hat = (2.0 / a2) * (m_dot + (1.0 - a1) * m_hat - 0.5 * a0) if abs(a2) > 1e-6 else m_hat ** 2

    return {
        'name': 'PINN-Analytical (chosen)',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics': True, 'full_supervised': False,
    }

class ThreeChannelPINN(nn.Module):
    OUTPUT_SCALE = 1.05

    def __init__(self, total_time=TOTAL_TIME, hidden=HIDDEN, depth=DEPTH):
        super().__init__()
        self.total_time = float(total_time)
        self.net = make_mlp(hidden=hidden, depth=depth, n_in=1, n_out=3)
        self.scalars = BoundedScalars()

    @property
    def a0(self): return self.scalars.a0
    @property
    def a1(self): return self.scalars.a1
    @property
    def a2(self): return self.scalars.a2

    def forward(self, t):
        out = self.OUTPUT_SCALE * torch.tanh(self.net(t / self.total_time))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

def three_channel_pinn(data, device='cpu', t_train_max=None):
    t_grid = data['t_grid']
    m_bar = data['m_bar']
    eps_bar = data['eps_bar']
    eps2_bar = data['eps2_bar']

    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t = torch.tensor(m_bar,dtype=torch.float32, device=device).view(-1, 1)
    e_t = torch.tensor(eps_bar, dtype=torch.float32, device=device).view(-1, 1)
    e2_t = torch.tensor(eps2_bar, dtype=torch.float32, device=device).view(-1, 1)
    train_idx, _ = _make_train_mask(t_grid, t_train_max, device)

    DATA_WEIGHT = 10.0

    torch.manual_seed(SEED)
    model = ThreeChannelPINN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def closure():
        opt.zero_grad()
        m_p, e_p, e2_p = model(t_t)
        L_d = (torch.mean((m_p[train_idx] - m_t[train_idx]) ** 2)
             + torch.mean((e_p[train_idx] - e_t[train_idx]) ** 2)
             + torch.mean((e2_p[train_idx] - e2_t[train_idx]) ** 2))

        tc = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        m_c, e_c, e2_c = model(tc)
        m_dot = time_derivative(m_c, tc)
        e_dot = time_derivative(e_c, tc)
        res_m = m_dot + (1.0 - model.a1) * m_c - 0.5 * model.a0 - 0.5 * model.a2 * e2_c
        res_e = e_dot + 2.0 * e_c - model.a1 - (model.a0 + model.a2) * m_c - model.a1 * e2_c
        L_p = torch.mean(res_m ** 2) + torch.mean(res_e ** 2)
        L = DATA_WEIGHT * L_d + L_p
        L.backward()
        return L

    t0 = time.time()
    adam_then_lbfgs(closure, list(model.parameters()),
                    n_adam=EPOCHS_ADAM, n_lbfgs=LBFGS_STEPS)
    train_time = time.time() - t0

    with torch.no_grad():
        m_p, e_p, e2_p = model(t_t)
        m_hat = m_p.cpu().numpy().squeeze()
        eps_hat = e_p.cpu().numpy().squeeze()
        eps2_hat= e2_p.cpu().numpy().squeeze()
        a0, a1, a2 = model.a0.item(), model.a1.item(), model.a2.item()

    return {
        'name': '3-channel PINN (fully supervised)',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics': True, 'full_supervised': True,
    }

def per_run_pinn(data_per_run, device='cpu', t_train_max=None, run_index=0):
    t_grid = data_per_run['t_grid']
    m_per_run = data_per_run['m_per_run']
    R, T = m_per_run.shape
    if run_index >= R:
        raise ValueError(f'run_index={run_index} but only {R} sub-runs available')
    m_one = m_per_run[run_index]

    t_t = torch.tensor(t_grid, dtype=torch.float32, device=device).view(-1, 1)
    m_t = torch.tensor(m_one, dtype=torch.float32, device=device).view(-1, 1)
    train_idx, _ = _make_train_mask(t_grid, t_train_max, device)

    torch.manual_seed(SEED)
    model = PINN_m_Core().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    DATA_WEIGHT = 10.0

    def closure():
        opt.zero_grad()
        L_d = torch.mean((model(t_t)[train_idx] - m_t[train_idx]) ** 2)
        tc = (torch.rand(N_COLLOC, 1, device=device) * TOTAL_TIME).requires_grad_(True)
        m_c = model(tc)
        m_dot = time_derivative(m_c, tc)
        res = m_dot + (1.0 - model.a1) * m_c - 0.5 * model.a0 - 0.5 * model.a2 * m_c ** 2
        L = DATA_WEIGHT * L_d + torch.mean(res ** 2)
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
    m_g = model(t_grad)
    m_dot = time_derivative(m_g, t_grad).detach().cpu().numpy().squeeze()
    eps2_hat = (2.0 / a2) * (m_dot + (1.0 - a1) * m_hat - 0.5 * a0) if abs(a2) > 1e-6 else m_hat ** 2

    return {
        'name': f'Per-run PINN (single sub-run, R[idx]={run_index})',
        'm_hat': m_hat, 'eps_hat': eps_hat, 'eps2_hat': eps2_hat,
        'a0': a0, 'a1': a1, 'a2': a2,
        'n_params': count_params(model), 'time_sec': train_time,
        'physics': True, 'full_supervised': False,
    }
