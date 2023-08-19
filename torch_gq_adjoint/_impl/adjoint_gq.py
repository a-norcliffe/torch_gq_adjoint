import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint, b_odeint       
# We include a new b_odeint, so that terms in the sum "drop out", solving
# up to a given t, then calculating the integrand and restarting the solve
# is slow compared to this method, since there is a fixed cost to restarting
# a solve. This way we solve past the point, and then "look backwards" to
# calculate it, independent of solving [y, a] backwards.
from .misc import _check_inputs, _flat_to_shape, _rms_norm, _mixed_linf_rms_norm, _wrap_norm



class OdeintAdjointGQMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, gtol, method, options, adjoint_rtol, adjoint_atol,
                adjoint_method, adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.gtol = gtol
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        with torch.no_grad():
            func.nfe = 0
            y = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, y, *adjoint_params)
        ctx.nfe = func.nfe
        return y

    @staticmethod
    def backward(ctx, grad_y):
        with torch.no_grad():
            shapes = ctx.shapes
            func = ctx.func
            nfe = ctx.nfe
            gtol = ctx.gtol
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            t, y, *adjoint_params = ctx.saved_tensors
            adjoint_params = tuple(adjoint_params)

            ##################################
            #     Set up adjoint_options     #
            ##################################

            if adjoint_options is None:
                adjoint_options = {}
            else:
                adjoint_options = adjoint_options.copy()

            # We assume that any grid points are given to us ordered in the same direction as for the forward pass (for
            # compatibility with setting adjoint_options = options), so we need to flip them around here.
            try:
                grid_points = adjoint_options['grid_points']
            except KeyError:
                pass
            else:
                adjoint_options['grid_points'] = grid_points.flip(0)

            # Backward compatibility: by default use a mixed L-infinity/RMS norm over the input, where we treat t, each
            # element of y, and each element of adj_y separately over the Linf, but consider all the parameters
            # together.
            if 'norm' not in adjoint_options:
                if shapes is None:
                    shapes = [y[-1].shape]  # [-1] because y has shape (len(t), *y0.shape)
                # y, adj_y corresponding to the order in aug_state below
                adjoint_shapes = 2*shapes
                adjoint_options['norm'] = _mixed_linf_rms_norm(adjoint_shapes)
            # ~Backward compatibility

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [y[-1], grad_y[-1]]  # y, adj_y

            ##################################
            #    Set up backward ODE func    #
            ##################################

            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[0]
                adj_y = y_aug[1]
                # ignore gradients wrt time and parameters

                with torch.enable_grad():
                    t_ = t.detach()
                    y = y.detach().requires_grad_(True)
                    func_eval = func(t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())
                    _y = torch.as_strided(y, (), ())
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)

                    vjp_y, = torch.autograd.grad(
                        func_eval, y, -adj_y,
                        allow_unused=True, retain_graph=False
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y

                return (func_eval, vjp_y)

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            grads = [torch.zeros_like(param) for param in adjoint_params]

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
                time_vjps[0] = 0
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    time_vjps[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time. And collect the integral values
                t1 = t[i]
                t0 = t[i-1]
                nterms = int(min(torch.ceil(gtol*nfe*(t1-t0)/(t[-1] - t[0])).item(), 64))
                """
                calculate the number of terms for the quadrature calculation, finds n based on
                the NFE. time and user tolerance, formula found empirically. Uses a formula for the
                error. Has a maximum of 64 terms, would work on a degree 127 polynomial
                which should be large enough for most applications.
                """
                aug_state, dgrads_param, dgrads_t = b_odeint(augmented_dynamics, tuple(aug_state),
                        t0, t1, nterms, func, adjoint_params, t_requires_grad, rtol=adjoint_rtol, atol=adjoint_atol,
                        method=adjoint_method, options=adjoint_options)
                
                grads = [g + dg for g, dg in zip(grads, dgrads_param)]

                if t_requires_grad:
                    time_vjps[0] += dgrads_t

                aug_state = [a[0] for a in aug_state]  # make aug_state correct dimensionality
                aug_state[0] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[1] += grad_y[i - 1]  # update any gradients wrt state at this time point

        return (None, None, aug_state[1], time_vjps, None, None, None, None, None, None, None, None, None, None, *grads)


def odeint_adjoint_gq(func, y0, t, rtol=1e-7, atol=1e-9, gtol=0.1, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument. If there are no parameters '
                         'then it is allowable to set `adjoint_params=()`.')

    # Must come before _check_inputs as we don't want to use normalised input (in particular any changes to options)
    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method
    if adjoint_options is None:
        adjoint_options = {k: v for k, v in options.items() if k != "norm"} if options is not None else {}
    if adjoint_params is None:
        adjoint_params = tuple(func.parameters())

    # Normalise to non-tupled input
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS)

    if "norm" in options and "norm" not in adjoint_options:
        adjoint_shapes = [y0.shape, y0.shape]
        adjoint_options["norm"] = _wrap_norm([_rms_norm, options["norm"], options["norm"]], adjoint_shapes)

    solution = OdeintAdjointGQMethod.apply(shapes, func, y0, t, rtol, atol, gtol, method, options, adjoint_rtol, adjoint_atol,
                                        adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution
