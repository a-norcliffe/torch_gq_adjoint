import abc
import torch
from .misc import _handle_unused_kwargs
from .misc import _flat_to_shape


def calc_vjps(func, t, aug_state, adjoint_params, t_requires_grad):
    """
    Calculates the vector jacobian products required to calculate the adjoint integral,
    i.e. the integrands at the prespecified points for quadrature calculation.
    """
    with torch.enable_grad():
        y = aug_state[0].squeeze(0)
        adj_y = aug_state[1].squeeze(0)
        t_ = -t.detach().to(y.dtype)  # have a minus because it goes backward in time, so all times are multiplied by -1
        t = t_.requires_grad_(True)
        y = y.detach()
        func_eval = func(t if t_requires_grad else t_, y)
        
        # calculate the vjps
        *vjp_params, vjp_t = torch.autograd.grad(
                        func_eval, [*adjoint_params, t], adj_y,
                        allow_unused=True, retain_graph=False)

        # autograd.grad returns None if no gradient, set to zero.
        vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
        vjp_params = [torch.zeros_like(param) if vjp_params is None else vjp_params
                        for param, vjp_params in zip(adjoint_params, vjp_params)]

    return vjp_params, vjp_t


class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def __init__(self, dtype, y0, norm, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.y0 = y0
        self.dtype = dtype

        self.norm = norm

    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    @abc.abstractmethod
    def _b_advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            solution[i] = self._advance(t[i])
        return solution

    def b_integrate(self, t, w_gq, forward_func, adjoint_params, t_requires_grad, shapes):
        # function will integrate aud_state all the way back and give corresponding
        # integrals of adj_param and adj_t
        t = t.to(self.dtype)
        self._before_integrate(t)

        # set up running sums for each integral, starting at 0
        grads_temp_param = [torch.zeros_like(param) for param in adjoint_params]
        if t_requires_grad:
            grads_temp_t = torch.zeros_like(t[0])
        else:
            grads_temp_t = None

        # ode solve aug_state to specified time and add the weighted term to running sum
        for i in range(1, len(t)-1):
            aug_state = self._b_advance(t[i])
            aug_state = _flat_to_shape(aug_state, (1,), shapes)
            dgrads_param, dgrads_t = calc_vjps(forward_func, t[i], aug_state, adjoint_params, t_requires_grad)
            grads_temp_param = [g + w_gq[i-1]*dg for g, dg in zip(grads_temp_param, dgrads_param)]
            if t_requires_grad:
                grads_temp_t += w_gq[i-1]*dgrads_t
        
        # finish ode solve and return the values
        aug_state = self._b_advance(t[-1])
        return aug_state, grads_temp_param, grads_temp_t


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        unused_kwargs.pop('norm', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dtype = y0.dtype
        self.device = y0.device

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t, dt, y):
        pass

    @abc.abstractmethod
    def _b_step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self._step_func(self.func, t0, t1 - t0, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1

        return solution

    def b_integrate(self, t, w_gq, forward_func, adjoint_params, t_requires_grad, shapes):
        # function will integrate aud_state all the way back and give corresponding
        # integrals of adj_param and adj_t
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        # set up running sums for each integral, starting at 0
        grads_temp_param = [torch.zeros_like(param) for param in adjoint_params]
        if t_requires_grad:
            grads_temp_t = torch.zeros_like(t[0])
        else:
            grads_temp_t = None

        # ode solve aug_state to specified time and add the weighted term to running sum
        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self._b_step_func(self.func, t0, t1 - t0, y0)
            y1 = y0 + dy

            if j == len(t)-1 and t1 >=t[j]:
                aug_state = self._linear_interp(t0, t1, y0, y1, t[j]) #at the end, solve the ode to the end

            while j < len(t)-1 and t1 >= t[j]:      # have len(t)-1 because we don't have weights from end point
                aug_state = self._linear_interp(t0, t1, y0, y1, t[j])
                aug_state = _flat_to_shape(aug_state, (1,), shapes)
                dgrads_param, dgrads_t = calc_vjps(forward_func, t[j], aug_state, adjoint_params, t_requires_grad)
                grads_temp_param = [g + w_gq[j-1]*dg for g, dg in zip(grads_temp_param, dgrads_param)]
                if t_requires_grad:
                    grads_temp_t += w_gq[j-1]*dgrads_t
                j += 1

            y0 = y1

        return aug_state, grads_temp_param, grads_temp_t

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
