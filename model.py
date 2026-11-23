"""SEIR model with policy schedules, a COVID extension and a measles preset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt

Array = np.ndarray
RateFn = Callable[[float], float]


def step_change(base: float, t0: float, factor: float) -> RateFn:
    """r(t) = base for t < t0, factor*base for t >= t0."""
    def r(t: float) -> float:
        return base if t < t0 else factor * base
    return r

def smooth_sigmoid_change(base: float, t_mid: float, width: float, factor: float) -> RateFn:
    """Smooth logistic transition around t_mid; smaller width = steeper change."""
    k = 1.0 / max(width, 1e-9)
    A = (1.0 - 1.0 / max(factor, 1e-9))
    def r(t: float) -> float:
        return base * (1.0 - A / (1.0 + np.exp(-(t - t_mid) * k)))
    return r

def seasonal_sine(base: float, amplitude: float, period: float, phase: float = 0.0) -> RateFn:
    """r(t) = base * (1 + amplitude * sin(2π (t - phase)/period)), amplitude ∈ [0,1)."""
    amp = float(np.clip(amplitude, 0.0, 0.999))
    w = 2.0 * np.pi / period
    def r(t: float) -> float:
        return base * (1.0 + amp * np.sin(w * (t - phase)))
    return r


@dataclass
class SEIRParams:
    beta: float      # transmission/contact rate
    sigma: float     # E->I (1/incubation)
    gamma: float     # I->R (1/infectious)
    mu: float = 0.0  # natural birth/death
    alpha: float = 0.0  # disease mortality on I
    omega: float = 0.0  # waning R->S
    v_rate: Optional[RateFn] = None   # vaccination S->R
    q_rate: Optional[RateFn] = None   # isolation I->Q
    gamma_q: float = 0.0              # Q->R recovery
    beta_t: Optional[RateFn] = None   # time-varying β(t)

    def beta_of(self, t: float) -> float:
        return float(self.beta if self.beta_t is None else self.beta_t(t))

def seir_rhs(t: float, y: Array, p: SEIRParams, N: float) -> Array:
    """y=[S,E,I,R] or [S,E,I,R,Q]; Q has zero infectivity."""
    S, E, I, R = y[:4]
    Q = y[4] if y.size == 5 else 0.0
    mu, alpha, omega = p.mu, p.alpha, p.omega
    beta_t = p.beta_of(t)
    v_t = p.v_rate(t) if p.v_rate else 0.0
    q_t = p.q_rate(t) if p.q_rate else 0.0

    lam = beta_t * I / max(N, 1e-12)
    dS = (mu * N) - lam*S - v_t*S + omega*R - mu*S
    dE = lam*S - p.sigma*E - mu*E
    dI = p.sigma*E - p.gamma*I - mu*I - alpha*I - q_t*I
    dR = p.gamma*I + v_t*S - omega*R - mu*R
    if y.size == 5:
        dQ = q_t*I - p.gamma_q*Q - mu*Q
        dR += p.gamma_q*Q
        return np.array([dS,dE,dI,dR,dQ])
    return np.array([dS,dE,dI,dR])

def R0_seir(p: SEIRParams) -> float:
    """ngm: R0 = (β σ) / ((σ+μ)(γ+μ)).  μ→0 ⇒ β/γ."""
    return (p.beta * p.sigma) / ((p.sigma + p.mu) * (p.gamma + p.mu))

def Rt_instant(t: float, state: Array, p: SEIRParams, N: float) -> float:
    """Rt ≈ R0 * (β(t)/β) * S/N."""
    S = state[0]
    return R0_seir(p) * (p.beta_of(t)/max(p.beta,1e-12)) * (S/max(N,1e-12))

def simulate_seir(
    T: Tuple[float,float],
    y0: Dict[str,float],
    p: SEIRParams,
    atol: float=1e-8, rtol: float=1e-6,
    max_step: Optional[float]=None,
    use_dynamic_N: bool=False,
) -> Dict[str,Array]:
    has_Q = p.q_rate is not None and p.gamma_q > 0.0
    state0 = np.array([y0.get('S',0.0), y0.get('E',0.0), y0.get('I',0.0), y0.get('R',0.0)])
    if has_Q: state0 = np.concatenate([state0, [y0.get('Q',0.0)]])
    N0 = float(np.sum(state0))

    def rhs(t, y):
        y = np.maximum(y, 0.0)
        N_use = float(np.sum(y)) if use_dynamic_N else N0
        return seir_rhs(t, y, p, N_use)

    ms = np.inf if (max_step is None) else (max_step if (np.isfinite(max_step) and max_step>0) else np.inf)
    sol = solve_ivp(rhs, T, state0, method="RK45", dense_output=True, atol=atol, rtol=rtol, max_step=ms)
    t = sol.t
    Y = np.clip(sol.y.T, 0.0, None)

    out: Dict[str,Array] = {"t": t}
    labels = ["S","E","I","R"] + (["Q"] if has_Q else [])
    for k,lab in enumerate(labels): out[lab] = Y[:,k]
    N_series = np.array([float(np.sum(Y[i])) if use_dynamic_N else N0 for i in range(len(t))])
    out["N"] = N_series
    out["Rt"] = np.array([Rt_instant(tt, Y[i], p, N_series[i]) for i,tt in enumerate(t)])

    inc = []
    for i,tt in enumerate(t):
        S,E,I,R = Y[i,:4]
        lam = p.beta_of(tt) * I / max(N_series[i],1e-12)
        inc.append(lam*S)
    inc = np.array(inc)
    out["incidence"] = inc
    out["cum_incidence"] = cumulative_trapezoid(inc, t, initial=0.0)
    return out


@dataclass
class COVIDParams:
    beta: float
    beta_t: Optional[RateFn] = None
    kappa_P: float = 1.0
    kappa_I: float = 1.0
    kappa_A: float = 0.5
    kappa_Q: float = 0.1
    T_incubation: float = 5.2  # E+P
    T_presym: float = 2.0      # P
    T_inf_I: float = 7.0
    T_inf_A: float = 5.0
    p_asym: float = 0.3
    q_rate: Optional[RateFn] = None
    gamma_Q: float = 1/10
    v_rate: Optional[RateFn] = None
    ve: float = 0.0
    omega_R: float = 0.0
    mu: float = 0.0
    def beta_of(self, t: float) -> float:
        return float(self.beta if self.beta_t is None else self.beta_t(t))

def covid_rhs(t: float, y: Array, p: COVIDParams, N: float) -> Array:
    S,E,P,I,A,Q,R = y
    beta_t = p.beta_of(t)
    q_t = p.q_rate(t) if p.q_rate else 0.0
    v_t = p.v_rate(t) if p.v_rate else 0.0
    sigma = 1.0/max(p.T_incubation - p.T_presym, 1e-9)  # E->P
    tau   = 1.0/max(p.T_presym, 1e-9)                  # P->I/A
    gamma_I = 1.0/max(p.T_inf_I, 1e-9)
    gamma_A = 1.0/max(p.T_inf_A, 1e-9)
    gamma_Q = p.gamma_Q

    # Force of infection with relative infectiousness and leaky vaccine
    infectious = p.kappa_P*P + p.kappa_I*I + p.kappa_A*A + p.kappa_Q*Q
    lam = beta_t * infectious / max(N, 1e-12)

    mu = p.mu
    dS = (mu*N) - (1 - p.ve)*lam*S - v_t*S + p.omega_R*R - mu*S
    dE = (1 - p.ve)*lam*S - sigma*E - mu*E
    dP = sigma*E - tau*P - mu*P
    dI = (1 - p.p_asym)*tau*P - (gamma_I + q_t + mu)*I
    dA = p.p_asym*tau*P - (gamma_A + mu)*A
    dQ = q_t*I - (gamma_Q + mu)*Q
    dR = gamma_I*I + gamma_A*A + gamma_Q*Q + v_t*S - mu*R - p.omega_R*R
    return np.array([dS,dE,dP,dI,dA,dQ,dR])

def R0_covid(p: COVIDParams) -> float:
    """Approx R0 (q≈0, mu≈0) using infectiousness-weighted mean durations."""
    q0 = 0.0
    if p.q_rate:
        try: q0 = float(p.q_rate(0.0))
        except: q0 = 0.0
    eff_T_I = 1.0 / (1.0/p.T_inf_I + q0)  # competing hazards approx
    return p.beta * (p.kappa_P*p.T_presym + (1-p.p_asym)*p.kappa_I*eff_T_I + p.p_asym*p.kappa_A*p.T_inf_A)

def simulate_covid(
    T: Tuple[float,float], y0: Dict[str,float], p: COVIDParams,
    atol: float=1e-8, rtol: float=1e-6, max_step: Optional[float]=None
) -> Dict[str,Array]:
    state0 = np.array([y0.get(k,0.0) for k in ("S","E","P","I","A","Q","R")])
    N0 = float(np.sum(state0))

    def rhs(t,y): return covid_rhs(t, np.maximum(y,0.0), p, N0)

    ms = np.inf if (max_step is None) else (max_step if (np.isfinite(max_step) and max_step>0) else np.inf)
    sol = solve_ivp(rhs, T, state0, method='RK45', dense_output=True, atol=atol, rtol=rtol, max_step=ms)
    t = sol.t
    Y = np.clip(sol.y.T, 0.0, None)
    comps = ["S","E","P","I","A","Q","R"]
    out: Dict[str,Array] = {"t": t}
    for i,c in enumerate(comps): out[c] = Y[:,i]
    out["N"] = Y.sum(axis=1)

    # Effective Rt includes leaky vaccine factor (1 - ve)
    out["Rt"] = np.array([
        (1.0 - p.ve) * (p.beta_of(tt)/p.beta) * (Y[i,0]/out["N"][i]) * R0_covid(p)
        for i, tt in enumerate(t)
    ])

    inc = []
    for i,tt in enumerate(t):
        S,E,P,I,A,Q,R = Y[i]
        infectious = p.kappa_P*P + p.kappa_I*I + p.kappa_A*A + p.kappa_Q*Q
        lam = p.beta_of(tt) * infectious / max(out["N"][i],1e-12)
        inc.append((1.0 - p.ve) * lam * S)
    inc = np.array(inc)
    out["incidence"] = inc
    out["cum_incidence"] = cumulative_trapezoid(inc, t, initial=0.0)
    return out


def summarize_epidemic(res: Dict[str,Array], primary_comp: str="I") -> Dict[str,float]:
    t, inc, cum, prim = res["t"], res["incidence"], res["cum_incidence"], res.get(primary_comp)
    if prim is None: raise KeyError(f"{primary_comp} not in results.")
    i_peak, p_peak = int(np.argmax(inc)), int(np.argmax(prim))
    return {
        "peak_incidence": float(inc[i_peak]),
        "t_peak_incidence": float(t[i_peak]),
        "peak_primary": float(prim[p_peak]),
        "t_peak_primary": float(t[p_peak]),
        "final_size": float(cum[-1]),
    }


def plot_dashboard(
    res: dict,
    *,
    title: str | None = None,
    show_Q: bool = True,
    annotate: bool = True,
    fraction: bool = True,      # plot as fractions of N by default
    per_100k: bool = False,     # or scale per 100k
    hide_S: bool = True,        # hide S so E/I/R are visible
    logy_traj: bool = False,    # optional log y for trajectories
) -> None:
    """2×2 dashboard with visibility controls.
    If fraction=True: y-axes are formatted as percentages (no scientific notation).
    Otherwise: y-axes use plain numbers (no scientific notation).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    t = res["t"]
    N_series = res.get("N", None)
    N = float(N_series[0]) if N_series is not None else None

    def scale(arr):
        x = np.asarray(arr, float)
        if fraction and N is not None:
            x = x / N
        if per_100k and N is not None:
            x = x * (1e5 if fraction else 1e5 / N)
        return x

    def y_label(base: str) -> str:
        if fraction:  return base + " (fraction)"
        if per_100k:  return base + " (per 100k)"
        return base

    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax_traj, ax_rt, ax_inc, ax_cum = axs.ravel()

    # Trajectories
    S = res.get("S", res.get("s"))
    E = res.get("E", res.get("e"))
    I = res.get("I", res.get("i"))
    R = res.get("R", res.get("r"))
    Q = res.get("Q", None)
    if S is not None and not hide_S: ax_traj.plot(t, scale(S), label="S")
    if E is not None:                 ax_traj.plot(t, scale(E), label="E")
    if I is not None:                 ax_traj.plot(t, scale(I), label="I/i")
    if R is not None:                 ax_traj.plot(t, scale(R), label="R")
    if show_Q and Q is not None:      ax_traj.plot(t, scale(Q), label="Q")
    ax_traj.set_title("Trajectories")
    ax_traj.set_xlabel("Time")
    ax_traj.set_ylabel(y_label("Population"))
    if logy_traj: ax_traj.set_yscale("log")
    ax_traj.legend(ncol=3)

    if "Rt" in res:
        rt = np.asarray(res["Rt"], float)
        ax_rt.plot(t, rt, label="R(t)")
        ax_rt.axhline(1.0, ls="--", lw=1, label="Rt = 1")
        ax_rt.set_title("Effective reproduction number")
        ax_rt.set_xlabel("Time"); ax_rt.set_ylabel("R(t)"); ax_rt.legend()
        if annotate and rt.size:
            below = np.where(rt < 1.0)[0]
            if below.size:
                idx = below[0]
                ax_rt.plot(t[idx], rt[idx], "o")
                ax_rt.annotate(f"Rt<1 at t≈{t[idx]:.1f}", (t[idx], rt[idx]),
                               xytext=(5, 10), textcoords="offset points")
    else:
        ax_rt.text(0.5, 0.5, "Rt not available", ha="center", va="center")
        ax_rt.set_axis_off()

    # Incidence
    if "incidence" in res:
        inc = np.asarray(res["incidence"], float)
        if fraction and N is not None: inc = inc / N
        if per_100k and N is not None: inc = inc * (1e5 if fraction else 1e5 / N)
        ax_inc.plot(t, inc, label="Incidence")
        ax_inc.set_title("Daily incidence")
        ax_inc.set_xlabel("Time")
        ax_inc.set_ylabel("Per day (fraction)" if fraction else ("Per day (per 100k)" if per_100k else "New infections per day"))
        ax_inc.legend()
        if annotate and inc.size:
            i_peak = int(np.argmax(inc))
            ax_inc.plot(t[i_peak], inc[i_peak], "o")
            ax_inc.annotate(f"Peak={inc[i_peak]:.3g} at t≈{t[i_peak]:.1f}",
                            (t[i_peak], inc[i_peak]), xytext=(5, 10), textcoords="offset points")
    else:
        ax_inc.text(0.5, 0.5, "Incidence not available", ha="center", va="center")
        ax_inc.set_axis_off()

    # Cumulative incidence
    if "cum_incidence" in res:
        ci = np.asarray(res["cum_incidence"], float)
        if fraction and N is not None: ci = ci / N
        if per_100k and N is not None: ci = ci * (1e5 if fraction else 1e5 / N)
        ax_cum.plot(t, ci, label="Cumulative incidence")
        ax_cum.set_title("Cumulative incidence (final size)")
        ax_cum.set_xlabel("Time")
        ax_cum.set_ylabel(y_label("Cumulative infections"))
        ax_cum.legend()
        if annotate and ci.size:
            ax_cum.plot(t[-1], ci[-1], "o")
            ax_cum.annotate(f"Final={ci[-1]:.3g}", (t[-1], ci[-1]),
                            xytext=(5, 10), textcoords="offset points")
    else:
        ax_cum.text(0.5, 0.5, "Cumulative incidence not available", ha="center", va="center")
        ax_cum.set_axis_off()

    if fraction:
        # Show percentages for all fraction plots
        fmt = PercentFormatter(xmax=1.0)
        for ax in (ax_traj, ax_inc, ax_cum):
            ax.yaxis.set_major_formatter(fmt)
    else:
        # Force plain numbers (no scientific notation)
        for ax in (ax_traj, ax_inc, ax_cum):
            ax.ticklabel_format(style="plain", axis="y")

    if title:
        fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    N = 1_000_000
    y0 = {"S": N-10.0, "E": 5.0, "I": 5.0, "R": 0.0}
    beta, sigma, gamma, mu = 0.30, 1/5.2, 1/7.0, 0.0
    beta_time = step_change(base=beta, t0=40.0, factor=0.5)  # simple NPI
    seir_p = SEIRParams(beta=beta, sigma=sigma, gamma=gamma, mu=mu, beta_t=beta_time)
    print(f"[SEIR] R0 = {R0_seir(seir_p):.3f}")
    res_seir = simulate_seir((0, 200), y0, seir_p, use_dynamic_N=True)
    plot_dashboard(res_seir, title="SEIR — Dashboard", fraction=True, hide_S=True)
    print("[SEIR] Summary:", summarize_epidemic(res_seir, primary_comp="I"))

    y0_c = {"S": N-50, "E": 20.0, "P": 20.0, "I": 10.0, "A": 0.0, "Q": 0.0, "R": 0.0}
    beta0 = 0.28
    beta_tv = step_change(base=beta0, t0=40.0, factor=0.5)
    covid_p = COVIDParams(
        beta=beta0, beta_t=beta_tv,
        kappa_P=1.0, kappa_I=1.0, kappa_A=0.5, kappa_Q=0.1,
        T_incubation=5.2, T_presym=2.0, T_inf_I=7.0, T_inf_A=5.0,
        p_asym=0.30,
        q_rate=lambda t: 0.05 if t>=40 else 0.0, gamma_Q=1/10,
        v_rate=None, ve=0.8, omega_R=1/180, mu=0.0
    )
    print(f"[COVID] Approx R0 ≈ {R0_covid(covid_p):.2f}")
    res_c = simulate_covid((0, 180), y0_c, covid_p)
    plot_dashboard(res_c, title="COVID — Dashboard", fraction=True, hide_S=True)
    print("[COVID] Summary:", summarize_epidemic(res_c, primary_comp="I"))

    life_expect_years = 70.0
    mu_measles = 1.0 / (life_expect_years * 365.0)
    N_m = 1_000_000
    y0_m = {"S": N_m - 1_000.0, "E": 500.0, "I": 500.0, "R": 0.0}
    beta_base = 1.8
    sigma_m   = 1/8.0
    gamma_m   = 1/7.0
    beta_seasonal = seasonal_sine(base=beta_base, amplitude=0.2, period=365.0, phase=0.0)
    v_daily = 0.0
    seir_measles = SEIRParams(
        beta=beta_base, sigma=sigma_m, gamma=gamma_m,
        mu=mu_measles, alpha=0.0, omega=0.0,
        v_rate=(lambda t: v_daily) if v_daily>0 else None,
        q_rate=None, gamma_q=0.0,
        beta_t=beta_seasonal
    )
    print(f"[Measles-like] R0 (no seasonality approx) ≈ {R0_seir(seir_measles):.2f}")
    res_m = simulate_seir((0, 3*365), y0_m, seir_measles, use_dynamic_N=False, max_step=1.0)
    plot_dashboard(res_m, title="Measles — Dashboard", fraction=True, hide_S=True)
    print("[Measles-like] Summary:", summarize_epidemic(res_m, primary_comp="I"))


# This block saves any open matplotlib figures to PNG files so the script
# produces artifacts even in headless environments (no GUI backend).
try:
    import matplotlib
    # Use a non-interactive backend to avoid requiring a GUI
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    out_dir = os.path.join(os.path.dirname(__file__) or ".", "figures")
    os.makedirs(out_dir, exist_ok=True)
    nums = plt.get_fignums()
    for i, num in enumerate(nums, 1):
        fig = plt.figure(num)
        path = os.path.join(out_dir, f"figure_{i}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    if nums:
        print(f"[Saved {len(nums)} figure(s) to: {out_dir}]")
    else:
        print("[No figures to save]")
except Exception as e:
    print("[Figure save skipped due to error]", e)
