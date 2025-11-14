from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from einops import rearrange, repeat, reduce

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Use 'notebook' instead for interactive plots

# import matplotlib_inline.backend_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats('retina')

# /cpfs01/user/liuxiaoran/miniconda3/envs/llm-ssm/

import seaborn as sns
sns.set(rc={
    "figure.dpi":300,
    'savefig.dpi':300,
    'animation.html':'jshtml',
    'animation.embed_limit':100, # Max animation size in Mb
})
sns.set_context('notebook')
sns.set_style('whitegrid') # or 'ticks'

# Utility functions for performing matrix scans faster
# The HiPPO functions also have pedagogical sequential versions, so this cell is not required

def shift_up(a, s=None, drop=True, dim=0):
    assert dim == 0
    if s is None:
        s = torch.zeros_like(a[0, ...])
    s = s.unsqueeze(dim)
    if drop:
        a = a[:-1, ...]
    return torch.cat((s, a), dim=dim)


def batch_mult(A, u, has_batch=None):
    """ Matrix mult A @ u with special case to save memory if u has additional batch dim

    The batch dimension is assumed to be the second dimension
    A : (L, ..., N, N)
    u : (L, [B], ..., N)
    has_batch: True, False, or None. If None, determined automatically

    Output:
    x : (L, [B], ..., N)
      A @ u broadcasted appropriately
    """

    if has_batch is None:
        has_batch = len(u.shape) >= len(A.shape)

    if has_batch:
        u = u.permute([0] + list(range(2, len(u.shape))) + [1])
    else:
        u = u.unsqueeze(-1)
    v = (A @ u)
    if has_batch:
        v = v.permute([0] + [len(u.shape)-1] + list(range(1, len(u.shape)-1)))
    else:
        v = v[..., 0]
    return v


def interleave(a, b, uneven=False, dim=0):
    """ Interleave two tensors of same shape """
    # assert(a.shape == b.shape)
    assert dim == 0 # TODO temporary to make handling uneven case easier
    if dim < 0:
        dim = N + dim
    if uneven:
        a_ = a[-1:, ...]
        a = a[:-1, ...]
    c = torch.stack((a, b), dim+1)
    out_shape = list(a.shape)
    out_shape[dim] *= 2
    c = c.view(out_shape)
    if uneven:
        c = torch.cat((c, a_), dim=dim)
    return c


def variable_unroll_general_sequential(A, u, s, op, variable=True):
    """ Unroll with variable (in time/length) transitions A with general associative operation

    A : ([L], ..., N, N) dimension L should exist iff variable is True
    u : (L, [B], ..., N) updates
    s : ([B], ..., N) start state
    output : x (..., N)
    x[i, ...] = A[i]..A[0] s + A[i..1] u[0] + ... + A[i] u[i-1] + u[i]
    """

    if not variable:
        A = A.expand((u.shape[0],) + A.shape)

    outputs = []
    for (A_, u_) in zip(torch.unbind(A, dim=0), torch.unbind(u, dim=0)):
        s = op(A_, s)
        s = s + u_
        outputs.append(s)

    output = torch.stack(outputs, dim=0)
    return output


def variable_unroll_general(A, u, s, op, compose_op=None, sequential_op=None, variable=True, recurse_limit=16):
    """ Bottom-up divide-and-conquer version of variable_unroll.

    compose is an optional function that defines how to compose A without multiplying by a leaf u
    """

    if u.shape[0] <= recurse_limit:
        if sequential_op is None:
            sequential_op = op
        return variable_unroll_general_sequential(A, u, s, sequential_op, variable)

    if compose_op is None:
        compose_op = op

    uneven = u.shape[0] % 2 == 1
    has_batch = len(u.shape) >= len(A.shape)

    u_0 = u[0::2, ...]
    u_1 = u[1::2, ...]

    if variable:
        A_0 = A[0::2, ...]
        A_1 = A[1::2, ...]
    else:
        A_0 = A
        A_1 = A

    u_0_ = u_0
    A_0_ = A_0
    if uneven:
        u_0_ = u_0[:-1, ...]
        if variable:
            A_0_ = A_0[:-1, ...]

    u_10 = op(A_1, u_0_) # batch_mult(A_1, u_0_, has_batch)
    u_10 = u_10 + u_1
    A_10 = compose_op(A_1, A_0_)

    # Recursive call
    x_1 = variable_unroll_general(A_10, u_10, s, op, compose_op, sequential_op, variable=variable, recurse_limit=recurse_limit)

    x_0 = shift_up(x_1, s, drop=not uneven)
    x_0 = op(A_0, x_0) # batch_mult(A_0, x_0, has_batch)
    x_0 = x_0 + u_0


    x = interleave(x_0, x_1, uneven, dim=0) # For some reason this interleave is slower than in the (non-variable) unroll_recursive
    return x


def variable_unroll_matrix(A, u, s=None, variable=True, recurse_limit=16):
    if s is None:
        s = torch.zeros_like(u[0])
    has_batch = len(u.shape) >= len(A.shape)
    op = lambda x, y: batch_mult(x, y, has_batch)
    sequential_op = lambda x, y: batch_mult(x.unsqueeze(0), y.unsqueeze(0), has_batch)[0]
    matmul = lambda x, y: x @ y
    return variable_unroll_general(A, u, s, op, compose_op=matmul, sequential_op=sequential_op, variable=variable, recurse_limit=recurse_limit)


def transition(measure, N, **measure_args):

    # Legendre (translated)
    assert measure == 'legt'
    Q = np.arange(N, dtype=np.float64)
    R = (2*Q + 1) ** .5
    j, i = np.meshgrid(Q, Q)
    A = R[:, None] * np.where(i < j, (-1.)**(i-j), 1) * R[None, :]
    B = R[:, None]
    A = -A
    return A, B

def measure(method, c=0.0):
    assert method == 'legt'
    fn = lambda x: np.heaviside(x, 0.0) * np.heaviside(1.0-x, 0.0)
    fn_tilted = lambda x: np.exp(c*x) * fn(x)
    return fn_tilted

def basis(method, N, vals, c=0.0, truncate_measure=True):
    """
    vals: list of times (forward in time)
    returns: shape (T, N) where T is length of vals
    """
    assert method == 'legt'
    
    # Ensure input is float32 type
    vals = np.array(vals, dtype=np.float32)
    n_range = np.arange(N, dtype=np.float32)
    
    # Calculate Legendre polynomial
    eval_matrix = ss.eval_legendre(n_range[:, None], 2*vals-1).T
    eval_matrix = eval_matrix.astype(np.float32)  # Convert to float32
    
    # Calculate scaling factor
    scale_factor = (2*n_range + 1).astype(np.float32) ** 0.5  # (N,)
    signs = (-1.) ** n_range  # (N,)
    
    # Correct broadcasting: first broadcast scale_factor and signs
    scaling = (scale_factor * signs)[:, None]  # (N, 1)
    eval_matrix = eval_matrix * scaling.T  # (T, N) * (1, N)
    
    if truncate_measure:
        eval_matrix[measure(method)(vals) == 0.0] = 0.0
    
    # Convert to float32 tensor and apply exponential decay
    p = torch.tensor(eval_matrix, dtype=torch.float32)
    p *= torch.tensor(np.exp(-c*vals), dtype=torch.float32)[:, None]
    
    return p


class HiPPO(nn.Module):
    """ Linear time invariant x' = Ax + Bu """
    def __init__(self, N, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.method = method
        self.N = N
        self.dt = dt
        self.T = T
        self.c = c
        
        A, B = transition(method, N)
        A = A + np.eye(N)*c
        self.A = A
        self.B = B.squeeze(-1)
        self.measure_fn = measure(method)
        
        C = np.ones((1, N))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        dB = dB.squeeze(-1)

        self.register_buffer('dA', torch.Tensor(dA)) # (N, N)
        self.register_buffer('dB', torch.Tensor(dB)) # (N,)

        self.vals = np.arange(0.0, T, dt)
        self.eval_matrix = basis(self.method, self.N, self.vals, c=self.c) # (T/dt, N)
        self.measure = measure(self.method)(self.vals)


    def forward(self, inputs, fast=True):
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        inputs = inputs.unsqueeze(-1)
        u = inputs * self.dB # (length, ..., N)

        if fast:
            dA = repeat(self.dA, 'm n -> l m n', l=u.size(0))
            return variable_unroll_matrix(dA, u)
        
        c = torch.zeros(u.shape[1:]).to(inputs)
        cs = []
        for f in inputs:
            c = F.linear(c, self.dA) + self.dB * f
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c, evals=None): # TODO take in a times array for reconstruction
        """
        c: (..., N,) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L,)
        """
        if evals is not None:
            eval_matrix = basis(self.method, self.N, evals)
        else:
            eval_matrix = self.eval_matrix

        m = self.measure[self.measure != 0.0]

        c = c.unsqueeze(-1)
        y = eval_matrix.to(c) @ c
        return y.squeeze(-1).flip(-1)
    
class MultiDimHiPPO(nn.Module):
    """Multi-dimensional Linear time invariant x' = Ax + Bu"""
    def __init__(self, N, input_dim, method='legt', dt=1.0, T=1.0, discretization='bilinear', scale=False, c=0.0):
        super().__init__()
        self.method = method #Use Legendre measure
        self.N = N
        self.input_dim = input_dim
        self.dt = dt
        self.T = T
        self.c = c
        
        self.base_N = N // input_dim  # Number of states per feature dimension
        # self.token_num=0 #Record current k

        base_N = self.base_N
        A, B = transition(method, base_N)  # A: (base_N, base_N), Define state dynamics for this feature dimension
        #B: (base_N, 1), Define input signal effect on states
        A = A + np.eye(base_N) * c #Add c*I to increase stability

        # B maintains original size, because all dimensions use the same B
        B = B.squeeze(-1)  # Shape: (base_N,)

        self.A = A  # Shape: (base_N, base_N)
        self.B = B  # Shape: (base_N,)
        self.measure_fn = measure(method)
        
        # Modify discretization part
        C = np.ones((1, base_N))  # Use base_N instead
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B.reshape(base_N, -1), C, D), dt=dt, method=discretization)
        dB = dB.reshape(base_N)  # Keep only base_N size, because all dimensions share same B

        # Convert to float32 and register buffers
        self.register_buffer('dA', torch.tensor(dA, dtype=torch.float32))  # (base_N, base_N) Save discretized matrix as non-trainable
        self.register_buffer('dB', torch.tensor(dB, dtype=torch.float32))  # (base_N,) Save discretized matrix as non-trainable

        self.vals = np.arange(0.0, T, dt)
        # eval_matrix also needs to use base_N
        self.eval_matrix = basis(self.method, base_N, self.vals, c=self.c)  # (T/dt, base_N)
        self.measure = measure(self.method)(self.vals)

    def forward(self, base_input, token_num, slice_input=None, fast=False):
        # base_input ( batchsize, base_N, input_dim ) 
        batch_size = base_input.size(0)
        print(f"Current self.token_num: {token_num}")
        if slice_input is not None:
            # token_num+=1
            base_N = self.N // self.input_dim // 2 # Number of states per dimension
            # print(base_input.shape)
            vals = np.ones(slice_input.shape[1], dtype=np.float32)*self.dt*token_num
            n_range = np.arange(base_N, dtype=np.float32)
            # print(vals.shape,n_range.shape)
            B = (vals[:, None] * n_range[None, :]) * 2 * np.pi / self.T
            B = np.stack([np.cos(B) / self.T, np.sin(B) / self.T], axis=-1).reshape((B.shape[0], -1))
            B = torch.from_numpy(B).to('cuda')

            print(f'B.shape:{B.shape}')

            C_add = torch.einsum('bln,ld->bdn', slice_input, B)   
            
            return base_input+C_add
        
        else:
            base_N = self.N // self.input_dim // 2 # Number of states per dimension
            # print(base_input.shape)
            vals = np.arange(base_input.shape[1], dtype=np.float32)*self.dt
            n_range = np.arange(base_N, dtype=np.float32)
            # print(vals.shape,n_range.shape)
            B = (vals[:, None] * n_range[None, :]) * 2 * np.pi / self.T
            B = np.stack([np.cos(B) / self.T, np.sin(B) / self.T], axis=-1).reshape((B.shape[0], -1))
            B = torch.from_numpy(B).to('cuda')

            print(f'B.shape:{B.shape}')

            C = torch.einsum('bln,ld->bdn', base_input, B)

            # self.token_num+=base_input.shape[1]

            return C  

    def reconstruct(self, c, clip_len,evals=None):
        """
        c: (..., N) HiPPO coefficients (same as x(t) in S4 notation)
        output: (..., L, input_dim)
        """
        base_N = self.N // self.input_dim // 2  # Number of states per dimension
        
        vals = np.arange(clip_len, dtype=np.float32)*self.dt
        n_range = np.arange(base_N, dtype=np.float32)
        B_inv = (vals[:, None] * n_range[None, :]) * 2 * np.pi / self.T
        B_inv = np.stack([np.cos(B_inv) / self.T, np.sin(B_inv) / self.T], axis=-1).reshape((clip_len, -1))
        B_inv = torch.from_numpy(B_inv).to('cuda')

        u_inv = torch.einsum('bdn,ld->bln', c, B_inv) / base_N

        # u_inv = (u_inv - u_inv.mean(axis=1, keepdims=True)) / u_inv.max(axis=1, keepdims=True)
        # max_vals = u_inv.max(axis=1, keepdims=True).values
        # u_inv = (u_inv - u_inv.mean(axis=1, keepdims=True)) / max_vals  (do it outside)
        print(u_inv.shape)
        return u_inv
### Synthetic data generation

def whitesignal(period, dt, freq, rms=0.5, batch_shape=()):
    
    period, dt  = period * 5, dt * 5
    """
    Produces output signal of length period / dt, band-limited to frequency freq
    Output shape (*batch_shape, period/dt)
    Adapted from the nengo library
    """

    if freq is not None and freq < 1. / period:
        raise ValueError(f"Make ``{freq=} >= 1. / {period=}`` to produce a non-zero signal",)

    nyquist_cutoff = 0.5 / dt
    if freq > nyquist_cutoff:
        raise ValueError(f"{freq} must not exceed the Nyquist frequency for the given dt ({nyquist_cutoff:0.3f})")

    n_coefficients = int(np.ceil(period / dt / 2.))
    shape = batch_shape + (n_coefficients + 1,)
    sigma = rms * np.sqrt(0.5)
    coefficients = 1j * np.random.normal(0., sigma, size=shape)
    coefficients[..., -1] = 0.
    coefficients += np.random.normal(0., sigma, size=shape)
    coefficients[..., 0] = 0.

    set_to_zero = np.fft.rfftfreq(2 * n_coefficients, d=dt) > freq
    coefficients *= (1-set_to_zero)
    power_correction = np.sqrt(1. - np.sum(set_to_zero, dtype=float) / n_coefficients)
    if power_correction > 0.: coefficients /= power_correction
    coefficients *= np.sqrt(2 * n_coefficients)
    signal = np.fft.irfft(coefficients, axis=-1)
    signal = signal - signal[..., :1]  # Start from 0
    return signal

# Modified whitesignal function for multi-dimensional signals
def multi_dim_whitesignal(period, dt, freq, dims=1, rms=0.5, batch_shape=()):
    """
    Produces multi-dimensional output signal
    Output shape (*batch_shape, period/dt, dims)
    """
    signals = []
    for _ in range(dims):
        signal = whitesignal(period, dt, freq, rms, batch_shape)
        signals.append(signal)
    return np.stack(signals, axis=-1).astype(np.float32)  # Ensure return float32


visualize = False
if visualize:
    x = whitesignal(1., high=3.0, dt=0.01, batch_shape=(2,))
    print(x.shape)
    plt.plot(np.arange(x.shape[-1]), x[0])
    plt.plot(np.arange(x.shape[-1]), x[1])
    plt.show()

# Animation code from "How to Train Your HiPPO"

sns.set_style('ticks')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plt_lines(x, y, color, size, label=None):
    return plt.plot(x, y, color, linewidth=size, label=label)[0]

def update_lines(ln, x, y):
    ln.set_data(x, y)

def animate_hippo(
    method, 
    T=1.0, dt=5e-4, N=64, freq=20.0,
    interval=100,
    plot_hippo=False, hippo_offset=0.0, label_hippo=False,
    plot_measure=False, measure_offset=-3.0, label_measure=False,
    plot_coeff=None, coeff_offset=3.0,
    plot_s4=False, s4_offset=6.0,
    plot_hippo_type='line', plot_measure_type='line', plot_coeff_type='line',
    size=1.0,
    plot_legend=True, plot_xticks=True, plot_box=True,
    plot_vline=False,
    animate_u=False,
    seed=2,
):
    np.random.seed(seed)

    vals = np.arange(0, int(T/dt)+1)
    L = int(T/dt)+1

    u = torch.FloatTensor(whitesignal(T, dt, freq=freq))
    u = F.pad(u, (1, 0))
    u = u + torch.FloatTensor(np.sin(1.5*np.pi/T*np.arange(0, T+dt, dt))) # add 3/4 of a sin cycle
    u = u.to(device)

    # hippo = HiPPOScale(method=method, N=N, max_length=L).to(device)
    hippo = HiPPO(method=method, N=N, dt=dt, T=T).to(device)
    coef_hippo = hippo(u).cpu().numpy()
    h_hippo = hippo.reconstruct(hippo(u)).cpu().numpy()
    u = u.cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 4))
    
    if animate_u:
        ln_u = plt_lines([], [], 'k', size, label='Input $u(t)$')
    else:
        plt_lines(vals, u, 'k', size, label='Input $u(t)$')
    
    if plot_hippo:
        label_args = {'label': 'HiPPO reconstruction'} if label_hippo else {}
        ln = plt_lines([], [], size=size, color='red', **label_args)

    if plot_measure:
        label_args = {'label': 'HiPPO Measure'} if label_measure else {}
        ln_measure = plt_lines(vals, np.zeros(len(vals))+measure_offset, size=size, color='green', **label_args)

    if plot_coeff is None: plot_coeff = []
    if isinstance(plot_coeff, int): plot_coeff = [plot_coeff]
    if len(plot_coeff) > 0:
        ln_coeffs = [
            plt_lines([], [], size=size, color='blue')
            for _ in plot_coeff
        ]
        plt_lines([], [], size=size, color='blue', label='State $x(t)$') # For the legend
        
    
    ### Y AXIS LIMITS
    if plot_measure:
        min_y = measure_offset
    else:
        min_y = np.min(u)
        
    if len(plot_coeff) > 0:
        max_u = np.max(u) + coeff_offset
    else:
        max_u = np.max(u)


    C = np.random.random(N)
    s4 = np.sum(coef_hippo * C, axis=-1)
    max_s4 = 0.0
    if plot_s4:
        ln_s4 = plt_lines([], [], size=size, color='red', label='Output $y(t)$')
        max_s4 = np.max(s4)+s4_offset
    
    if plot_vline:
        ln_vline = ax.axvline(0, ls='-', color='k', lw=1)

    if plot_legend:
        plt.legend(loc='upper left', fontsize='x-small')


    def init():
        left_endpoint = vals[0]
        ax.set_xlim(left_endpoint, vals[-1]+1)
        ax.set_ylim(min_y, max(max_u, max_s4))
        ax.set_yticks([])
        if not plot_xticks: ax.set_xticks([])
        if not plot_box: plt.box(False)
        return [] # ln,

    def update(frame):
        if animate_u:
            xdata = np.arange(frame)
            ydata = u[:frame]
            update_lines(ln_u, xdata, ydata)

        m = np.zeros(len(vals))
        m[:frame] = hippo.measure_fn(np.arange(frame)*dt)[::-1]
        xdata = vals
        if plot_measure:
            update_lines(ln_measure, xdata, m+measure_offset)
        
        if plot_hippo:   
            ydata = h_hippo[frame] + hippo_offset
            m2 = hippo.measure_fn(np.arange(len(ydata))*dt)[::-1]
            # Remove reconstruction where measure is 0
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame-len(ydata), frame)
            update_lines(ln, xdata, ydata)

        if len(plot_coeff) > 0:
            for coeff, ln_coeff in zip(plot_coeff, ln_coeffs):
                update_lines(ln_coeff, np.arange(frame), coef_hippo[:frame, coeff] + coeff_offset)
        if plot_s4: # Only scale case; scale case should copy plot_hippo logic
            update_lines(ln_s4, np.arange(0, frame), s4[:frame] + s4_offset)
            
        if plot_vline:
            ln_vline.set_xdata([frame, frame])

        return []

    ani = FuncAnimation(fig, update,
                        frames=np.arange(0, int(T*1000/interval)+1)*int(interval/1000/dt),
                        interval=interval,
                        init_func=init, blit=True)

    return ani

def animate_multi_dim_hippo(
    method='legt',
    T=1.0, dt=1e-4, N=256, # N: Total state dimension count
    input_dim=2,  # Input dimension, each input dimension state is N//input_dim dimensions
    freq=20.0,
    interval=100,
    size=1.0,
    plot_measure=True,
    plot_vline=True,
    seed=2,
):
    """
    Animate multi-dimensional signal reconstruction
    """
    np.random.seed(seed)
    
    # Setup time points
    vals = np.arange(0, int(T/dt)+1)
    L = int(T/dt)+1
    
    # Modify data generation and transformation part
    signal = multi_dim_whitesignal(T, dt, freq=freq, dims=input_dim)
    signal = signal.astype(np.float32)
    u = torch.tensor(signal, dtype=torch.float32)
    u = F.pad(u, (0, 0, 1, 0))
    
    # Add sinusoidal components
    t = np.arange(0, T+dt, dt, dtype=np.float32)
    for dim in range(input_dim):
        sin_wave = np.sin((1.5 + dim*0.5)*np.pi/T*t).astype(np.float32)
        u[..., dim] = u[..., dim] + torch.tensor(sin_wave, dtype=torch.float32)
    
    # Initialize multi-dimensional HiPPO
    hippo = MultiDimHiPPO(
        method=method, 
        N=N, 
        input_dim=input_dim,
        dt=dt, 
        T=T
    )
    
    # Get reconstructions
    coef_hippo = hippo(u)
    h_hippo = hippo.reconstruct(coef_hippo).numpy()
    u = u.numpy()
    
    # Setup the figure
    fig, axs = plt.subplots(input_dim, 1, figsize=(12, 4*input_dim))
    if input_dim == 1:
        axs = [axs]
    
    # Setup lines for each dimension
    lines_dict = {}
    for dim in range(input_dim):
        ax = axs[dim]
        
        # Original signal
        lines_dict[f'u_{dim}'] = ax.plot(
            [], [], 'k', 
            linewidth=size, 
            label=f'Input $u_{dim}(t)$'
        )[0]
        
        # Reconstruction
        lines_dict[f'reconstruction_{dim}'] = ax.plot(
            [], [], 'r', 
            linewidth=size, 
            label=f'HiPPO reconstruction_{dim}'
        )[0]
        
        # Measure line if requested
        if plot_measure:
            lines_dict[f'measure_{dim}'] = ax.plot(
                vals, 
                np.zeros(len(vals))-3.0,
                'g',
                linewidth=size,
                label=f'HiPPO Measure_{dim}'
            )[0]
        
        # Vertical line if requested
        if plot_vline:
            lines_dict[f'vline_{dim}'] = ax.axvline(0, ls='-', color='k', lw=1)
        
        ax.legend(loc='upper left', fontsize='x-small')
        ax.set_title(f'Dimension {dim+1}')
    
    def init():
        left_endpoint = vals[0]
        for dim in range(input_dim):
            ax = axs[dim]
            ax.set_xlim(left_endpoint, vals[-1]+1)
            ax.set_ylim(np.min(u[..., dim])-3.0, np.max(u[..., dim])+1.0)
        return []
    
    def update(frame):
        ret = []
        for dim in range(input_dim):
            # Update input signal
            xdata = np.arange(frame)
            ydata = u[:frame, dim]
            lines_dict[f'u_{dim}'].set_data(xdata, ydata)
            
            # Update measure if requested
            if plot_measure:
                m = np.zeros(len(vals))
                m[:frame] = hippo.measure_fn(np.arange(frame)*dt)[::-1]
                lines_dict[f'measure_{dim}'].set_data(vals, m-3.0)
            
            # Update reconstruction
            ydata = h_hippo[frame, :, dim]
            m2 = hippo.measure_fn(np.arange(len(ydata))*dt)[::-1]
            ydata[m2 == 0.0] = np.nan
            xdata = np.arange(frame-len(ydata), frame)
            lines_dict[f'reconstruction_{dim}'].set_data(xdata, ydata)
            
            # Update vertical line if requested
            if plot_vline:
                lines_dict[f'vline_{dim}'].set_xdata([frame, frame])
            
        return []
    
    ani = FuncAnimation(
        fig, 
        update,
        frames=np.arange(0, int(T*1000/interval)+1)*int(interval/1000/dt),
        interval=interval,
        init_func=init, 
        blit=True
    )
    
    plt.tight_layout()
    return ani


# Example usage
def visualize_example():
    ani = animate_multi_dim_hippo(
        method='legt',
        T=1, 
        dt=1e-3,  # Increased dt for faster rendering
        N=64, 
        input_dim=3,  # 3D signal
        interval=100,
        size=1.0,
        plot_measure=True,
        plot_vline=True,
    )
    
    # Save animation
    ani.save('multi_dim_hippo.gif', writer='pillow', fps=10)
    
    return ani

   
# # Visualize HiPPO online reconstruction

# ani = animate_hippo(
#     'legt', # Try 'legt' or 'fourier', lagt, legs
#     T=1, dt=1e-5, N=256, interval=100,
#     # T=1, dt=1e-8, N=2048, interval=100,  # too slow, need fast
#     # T=1, dt=1e-3, N=48, interval=200, # Faster rendering for testing
#     size=1.0,

#     animate_u=True,
#     plot_hippo=True, hippo_offset=0.0, label_hippo=True,
#     plot_s4=False, s4_offset=6.0,
#     plot_measure=True, measure_offset=-3.0, label_measure=True,
#     plot_coeff=[], coeff_offset=3.0,
#     plot_legend=True, plot_xticks=True, plot_box=True,
#     plot_vline=True,
# )

# ani.save('/remote-home1/yrsong/move_LegT_241216b.gif', writer='imagemagick', fps=6)

# print('yes')

# Create animation
# ani = animate_multi_dim_hippo(
#     method='legt',
#     T=1, 
#     dt=1e-4,
#     N=512, 
#     input_dim=4,
#     interval=100,
#     size=1.0,
#     plot_measure=True,
#     plot_vline=True,
# )
# ani.save('/remote-home1/syhe/hippoattention/multi_dim_hippo.gif', writer='pillow', fps=10)

print('yes')