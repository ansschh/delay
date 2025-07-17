# solvers/method_of_steps.py

import numpy as np
from scipy.integrate import solve_ivp

def solve_dde(f, history, τ, t_span, dt, rtol=1e-6, atol=1e-8):
    """
    Method-of-steps:
      f: rhs u'(t) = f(t, u(t), u(t-τ))
      history: callable h(t) for t ≤ 0
      τ: delay
      t_span: (0, T)
      dt: step size for window Δt
    Returns times, solution array on a fine grid.
    """
    t0, Tf = t_span
    # initial condition over [-τ,0]
    H = history
    sol_t = []
    sol_y = []
    # how many windows?
    N = int(np.ceil((Tf - t0) / dt))
    # store past solution for interpolation
    from scipy.interpolate import interp1d
    past_t = np.linspace(-τ, 0, max(100, int(τ/dt)))
    past_y = np.vstack([H(ti) for ti in past_t])
    interp = interp1d(past_t, past_y, axis=0, fill_value="extrapolate")
    u0 = past_y[-1]

    for i in range(N):
        t_start = i*dt
        t_end   = min((i+1)*dt, Tf)
        def rhs(t, y):
            y_delay = interp(t-τ)
            return f(t, y, y_delay)
        sol = solve_ivp(rhs, (t_start, t_end), u0,
                        rtol=rtol, atol=atol,
                        dense_output=True)
        ts = sol.t
        ys = sol.y.T
        sol_t.append(ts)
        sol_y.append(ys)
        # update interpolant
        new_t = sol.t
        new_y = sol.y.T
        past_t = np.concatenate([past_t, new_t])
        past_y = np.vstack([past_y, new_y])
        interp = interp1d(past_t, past_y, axis=0, fill_value="extrapolate")
        u0 = new_y[-1]
    return np.concatenate(sol_t), np.vstack(sol_y)
