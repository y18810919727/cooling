import math
import torch
import numpy as np
def cheap_stack(tensors, dim):
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)
    else:
        return torch.stack(tensors, dim=dim)

def tridiagonal_solve(b, A_upper, A_diagonal, A_lower):
    """Solves a tridiagonal system Ax = b.

    The arguments A_upper, A_digonal, A_lower correspond to the three diagonals of A. Letting U = A_upper, D=A_digonal
    and L = A_lower, and assuming for simplicity that there are no batch dimensions, then the matrix A is assumed to be
    of size (k, k), with entries:

    D[0] U[0]
    L[0] D[1] U[1]
         L[1] D[2] U[2]                     0
              L[2] D[3] U[3]
                  .    .    .
                       .      .      .
                           .        .        .
                        L[k - 3] D[k - 2] U[k - 2]
           0                     L[k - 2] D[k - 1] U[k - 1]
                                          L[k - 1]   D[k]

    Arguments:
        b: A tensor of shape (..., k), where '...' is zero or more batch dimensions
        A_upper: A tensor of shape (..., k - 1).
        A_diagonal: A tensor of shape (..., k).
        A_lower: A tensor of shape (..., k - 1).

    Returns:
        A tensor of shape (..., k), corresponding to the x solving Ax = b

    Warning:
        This implementation isn't super fast. You probably want to cache the result, if possible.
    """

    # This implementation is very much written for clarity rather than speed.

    A_upper, _ = torch.broadcast_tensors(A_upper, b[..., :-1])
    A_lower, _ = torch.broadcast_tensors(A_lower, b[..., :-1])
    A_diagonal, b = torch.broadcast_tensors(A_diagonal, b)

    channels = b.size(-1)

    new_b = np.empty(channels, dtype=object)
    new_A_diagonal = np.empty(channels, dtype=object)
    outs = np.empty(channels, dtype=object)

    new_b[0] = b[..., 0]
    new_A_diagonal[0] = A_diagonal[..., 0]
    for i in range(1, channels):
        w = A_lower[..., i - 1] / new_A_diagonal[i - 1]
        new_A_diagonal[i] = A_diagonal[..., i] - w * A_upper[..., i - 1]
        new_b[i] = b[..., i] - w * new_b[i - 1]

    outs[channels - 1] = new_b[channels - 1] / new_A_diagonal[channels - 1]
    for i in range(channels - 2, -1, -1):
        outs[i] = (new_b[i] - A_upper[..., i] * outs[i + 1]) / new_A_diagonal[i]

    return torch.stack(outs.tolist(), dim=-1)


def _natural_cubic_spline_coeffs_with_missing_values(t, path):
    a_pieces = []
    b_pieces = []
    two_c_pieces = []
    three_d_pieces = []
    for p in path.unbind(dim=0):  # TODO: parallelise over this
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, p)
        a_pieces.append(a)
        b_pieces.append(b)
        two_c_pieces.append(two_c)
        three_d_pieces.append(three_d)
    return (cheap_stack(a_pieces, dim=0),
            cheap_stack(b_pieces, dim=0),
            cheap_stack(two_c_pieces, dim=0),
            cheap_stack(three_d_pieces, dim=0))


def _natural_cubic_spline_coeffs_without_missing_values(times, path):
    # path should be a tensor of shape (..., length)
    # Will return the b, two_c, three_d coefficients of the derivative of the cubic spline interpolating the path.

    length = path.size(-1)

    if length < 2:
        # In practice this should always already be caught in __init__.
        raise ValueError("Must have a time dimension of size at least 2.")
    elif length == 2:
        a = path[..., :1]
        b = (path[..., 1:] - path[..., :1]) / (times[..., 1:] - times[..., :1])
        two_c = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
        three_d = torch.zeros(*path.shape[:-1], 1, dtype=path.dtype, device=path.device)
    else:
        # Set up some intermediate values
        time_diffs = times[1:] - times[:-1]
        time_diffs_reciprocal = time_diffs.reciprocal()
        time_diffs_reciprocal_squared = time_diffs_reciprocal ** 2
        three_path_diffs = 3 * (path[..., 1:] - path[..., :-1])
        six_path_diffs = 2 * three_path_diffs
        path_diffs_scaled = three_path_diffs * time_diffs_reciprocal_squared

        # Solve a tridiagonal linear system to find the derivatives at the knots
        system_diagonal = torch.empty(length, dtype=path.dtype, device=path.device)
        system_diagonal[:-1] = time_diffs_reciprocal
        system_diagonal[-1] = 0
        system_diagonal[1:] += time_diffs_reciprocal
        system_diagonal *= 2
        system_rhs = torch.empty_like(path)
        system_rhs[..., :-1] = path_diffs_scaled
        system_rhs[..., -1] = 0
        system_rhs[..., 1:] += path_diffs_scaled
        knot_derivatives = tridiagonal_solve(system_rhs, time_diffs_reciprocal, system_diagonal,
                                             time_diffs_reciprocal)

        # Do some algebra to find the coefficients of the spline
        a = path[..., :-1]
        b = knot_derivatives[..., :-1]
        two_c = (six_path_diffs * time_diffs_reciprocal
                 - 4 * knot_derivatives[..., :-1]
                 - 2 * knot_derivatives[..., 1:]) * time_diffs_reciprocal
        three_d = (-six_path_diffs * time_diffs_reciprocal
                   + 3 * (knot_derivatives[..., :-1]
                          + knot_derivatives[..., 1:])) * time_diffs_reciprocal_squared

    return a, b, two_c, three_d

def natural_cubic_spline_coeffs(t, X): #三次样条系数
    if not t.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if not X.is_floating_point():
        raise ValueError("t and X must both be floating point/")
    if len(t.shape) != 1:
        raise ValueError("t must be one dimensional.")
    prev_t_i = -math.inf
    for t_i in t:
        if t_i <= prev_t_i:
            raise ValueError("t must be monotonically increasing.")

    if len(X.shape) < 2:
        raise ValueError("X must have at least two dimensions, corresponding to time and channels.")

    if X.size(-2) != t.size(0):
        raise ValueError("The time dimension of X must equal the length of t.")

    if t.size(0) < 2:
        raise ValueError("Must have a time dimension of size at least 2.")

    if torch.isnan(X).any():
        # Transpose because channels are a batch dimension for the purpose of finding interpolating polynomials.
        # b, two_c, three_d have shape (..., channels, length - 1)
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_with_missing_values(t, X.transpose(-1, -2))
    else:
        # Can do things more quickly in this case.
        a, b, two_c, three_d = _natural_cubic_spline_coeffs_without_missing_values(t, X.transpose(-1, -2))

    # These all have shape (..., length - 1, channels)
    a = a.transpose(-1, -2)
    b = b.transpose(-1, -2)
    two_c = two_c.transpose(-1, -2)
    three_d = three_d.transpose(-1, -2)
    return a, b, two_c, three_d

class NaturalCubicSpline:
    """Calculates the natural cubic spline approximation to the batch of controls given. Also calculates its derivative.

    Example:
        times = torch.linspace(0, 1, 7)
        # (2, 1) are batch dimensions. 7 is the time dimension (of the same length as t). 3 is the channel dimension.
        X = torch.rand(2, 1, 7, 3)
        coeffs = natural_cubic_spline_coeffs(times, X)
        # ...at this point you can save the coeffs, put them through PyTorch's Datasets and DataLoaders, etc...
        spline = NaturalCubicSpline(times, coeffs)
        t = torch.tensor(0.4)
        # will be a tensor of shape (2, 1, 3), corresponding to batch and channel dimensions
        out = spline.derivative(t)
    """

    def __init__(self, times, coeffs, **kwargs):
        """
        Arguments:
            times: As was passed as an argument to natural_cubic_spline_coeffs.
            coeffs: As returned by natural_cubic_spline_coeffs.
        """
        super(NaturalCubicSpline, self).__init__(**kwargs)

        a, b, two_c, three_d = coeffs

        self._times = times
        self._a = a
        self._b = b
        # as we're typically computing derivatives, we store the multiples of these coefficients that are more useful
        self._two_c = two_c
        self._three_d = three_d

    def _interpret_t(self, t):
        maxlen = self._b.size(-2) - 1
        index = (t > self._times).sum() - 1
        index = index.clamp(0, maxlen)  # clamp because t may go outside of [t[0], t[-1]]; this is fine
        # will never access the last element of self._times; this is correct behaviour
        fractional_part = t - self._times[index]
        return fractional_part, index

    def evaluate(self, t):
        """Evaluates the natural cubic spline interpolation at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = 0.5 * self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part / 3
        inner = self._b[..., index, :] + inner * fractional_part
        return self._a[..., index, :] + inner * fractional_part

    def derivative(self, t):
        """Evaluates the derivative of the natural cubic spline at a point t, which should be a scalar tensor."""
        fractional_part, index = self._interpret_t(t)
        inner = self._two_c[..., index, :] + self._three_d[..., index, :] * fractional_part
        deriv = self._b[..., index, :] + inner * fractional_part
        return deriv
