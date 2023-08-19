import torch
import numpy as np
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .misc import _check_inputs, _flat_to_shape

SOLVERS = {
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'adaptive_heun': AdaptiveHeunSolver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'explicit_adams': AdamsBashforth,
    'implicit_adams': AdamsBashforthMoulton,
    # Backward compatibility: use the same name as before
    'fixed_adams': AdamsBashforthMoulton,
    # ~Backwards compatibility
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time. 
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)
    solution = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


def b_odeint(func, aug_state, t0, t1, nterms, forward_func, adjoint_params, t_requires_grad, rtol=1e-7, atol=1e-9, method=None, options=None):
    """Integrate a system of ordinary differential equations, specifically the
    adjoint equations backwards. And solve for the adj_param and adj_t integrals

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        y = [x, a]
        dy/dt = func(t, y) = [f(t, x), -a*df/dx]
        y1 = [x1, a1]
        ```
    where x is a Tensor or tuple of Tensors of any shape and a is the adjoint
    with the same shape as x. Additionally the adjoint integrals are solved with Gaussian
    Quadrature. This is solved from t1 to t0 backwards in time

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors. Here it is [f, -a*d/dx]
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors. Here it is [x(T), a(T)]
        t0: 1-D Tensor holding the earlier time t0 
        t1: 1-D Tensor holding the later time t1 
        nterms: int that says how many terms are used in the Gaussian Quadrature calculation
        forward_func: the function f(t, x)
        adjoint_params: tuple of tensors, the parameters of the function
        t_requires_grad: bool saying whether the measurement times require gradient or not
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        gtol: optional upperbound on error of the estimated gradients
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """

    # set up the integration and the Legendre times and weights
    a_gq = (t1-t0)/2
    b_gq = (t1+t0)/2
    t_gq, w_gq = np.polynomial.legendre.leggauss(nterms)
    t_gq = torch.tensor(t_gq).flip(0).to(t0.device)
    w_gq = torch.tensor(w_gq).to(t0.device) # Don't need to flip they are symmetric.
    t_gq = a_gq*t_gq + b_gq
    w_gq = a_gq*w_gq

    t = torch.cat((t1.unsqueeze(0), t_gq, t0.unsqueeze(0)))
    shapes, func, aug_state, t, rtol, atol, method, options = _check_inputs(func, aug_state, t, rtol, atol, method, options, SOLVERS)

    solver = SOLVERS[method](func=func, y0=aug_state, rtol=rtol, atol=atol, **options)
    aug_state, grads_param, grads_t = solver.b_integrate(t, w_gq, forward_func, adjoint_params, t_requires_grad, shapes)

    if shapes is not None:
        aug_state = _flat_to_shape(aug_state, (1,), shapes) #len(t) becomes 1 because we only care about the final y and adj_y

    return aug_state, grads_param, grads_t
