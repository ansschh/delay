# solvers/radar5_wrapper.py

from pydelay import dde23

def solve_stiff_dde(eqns, params, τs, history, t_eval):
    """
    eqns: dict{'u': '... expression ...'}
    τs: list of delays in same order as used in eqns strings
    history: dict{'u': history_fn}
    t_eval: array of times at which to sample
    """
    dde = dde23(eqns=eqns, params=params, τ=τs)
    dde.set_sim_params(AbsTol=1e-8, RelTol=1e-6)
    dde.hist_from_funcs(history, 0.0)
    dde.run()
    sol = dde.sample(t_eval)
    return sol['t'], sol['u']
