"""
Rethinking Attention with Performers
https://arxiv.org/abs/2009.14794
"""
# https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from functools import partial

##

# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/local_attention.py

# class LocalAttention(nn.Module):
#     """Implement fast local attention where a query can only attend to
#     neighboring keys.
#     In this attention module the query Q_i can only attend to a key K_j if
#     |i-j| < local_context/2.
#     Arguments
#     ---------
#         local_context: The neighborhood to consider for local attention.
#         softmax_temp: The temperature to use for the softmax attention.
#                       (default: 1/sqrt(d_keys) where d_keys is computed at
#                       runtime)
#         attention_dropout: The dropout rate to apply to the attention
#                            (default: 0.1)
#         event_dispatcher: str or EventDispatcher instance to be used by this
#                           module for dispatching events (default: the default
#                           global dispatcher)
#     """
#     def __init__(self, local_context, softmax_temp=None, attention_dropout=0.1,
#                  event_dispatcher=""):
#         super(LocalAttention, self).__init__()
#         self.local_context = local_context
#         self.softmax_temp = softmax_temp
#         self.dropout = Dropout(attention_dropout)
#         self.event_dispatcher = EventDispatcher.get(event_dispatcher)

#     def forward(self, queries, keys, values, attn_mask, query_lengths,
#                 key_lengths):
#         """Implements the local attention.
#         The attn_mask can be anything but the only values that will be
#         considered will be the ones in the neighborhood of each query.
#         Arguments
#         ---------
#             queries: (N, L, H, E) The tensor containing the queries
#             keys: (N, S, H, E) The tensor containing the keys
#             values: (N, S, H, D) The tensor containing the values
#             attn_mask: An implementation of BaseMask that encodes where each
#                        query can attend to
#             query_lengths: An implementation of  BaseMask that encodes how
#                            many queries each sequence in the batch consists of
#             key_lengths: An implementation of BaseMask that encodes how
#                          many queries each sequence in the batch consists of
#         """
#         # Extract some shapes and compute the temperature
#         N, L, H, E = queries.shape
#         _, S, _, D = values.shape
#         context = self.local_context
#         softmax_temp = self.softmax_temp or 1./sqrt(E)

#         # Permute the dimensions to NHLE instead of NLHE
#         queries = queries.permute(0, 2, 1, 3).contiguous()
#         keys = keys.permute(0, 2, 1, 3).contiguous()
#         values = values.permute(0, 2, 1, 3).contiguous()

#         QK = local_dot_product(
#             queries,
#             keys,
#             attn_mask.additive_matrix_finite,
#             key_lengths.lengths,
#             self.local_context
#         )
#         A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))

#         V_new = local_weighted_average(A, values)

#         return V_new.permute(0, 2, 1, 3).contiguous()

##

import torch
import torch.nn as nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

# for routing arguments into the functions of the reversible layer
def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args

# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args = {}, g_args = {}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args = {}, g_args = {}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, args):
        ctx.args = args
        for block, kwarg in zip(blocks, args):
            x = block(x, **kwarg)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        args = ctx.args
        for block, kwargs in zip(ctx.blocks[::-1], args[::-1]):
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route = {}):
        super().__init__()
        assert all(len(route) == len(layers) for route in args_route.values()), 'each argument route map must have the same depth as the number of sequential layers'
        self.layers = layers
        self.args_route = args_route

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x

class ReversibleSequence(nn.Module):
    def __init__(self, blocks, args_route = {}):
        super().__init__()
        self.args_route = args_route
        self.blocks = nn.ModuleList([ReversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)

        blocks = self.blocks
        args = route_args(self.args_route, kwargs, len(blocks))
        args = list(map(lambda x: {'f_args': x[0], 'g_args': x[1]}, args))

        out =  _ReversibleFunction.apply(x, blocks, args)
        return torch.stack(out.chunk(2, dim=-1)).sum(dim=0)

## 

# helpers

def exists(val):
    return val is not None

def empty(tensor):
    return tensor.numel() == 0

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val):
    return (val,) if not isinstance(val, tuple) else val

def get_module_device(module):
    return next(module.parameters()).device

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_self_attention/fast_self_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime

def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.qr(unstructured_block.cpu(), some = True)
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# linear attention classes with softmax kernel

# non-causal linear attention
def linear_attention(q, k, v):
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k.sum(dim = -2))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# efficient causal linear attention, created by EPFL
# TODO: rewrite EPFL's CUDA kernel to do mixed precision and remove half to float conversion and back
def causal_linear_attention(q, k, v, amp_enabled = False):
    from fast_transformers.causal_product import CausalDotProduct
    is_half = isinstance(q, torch.cuda.HalfTensor) or amp_enabled

    if is_half:
        q, k, v = map(lambda t: t.float(), (q, k, v))

    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k.cumsum(dim=-2))
    out = CausalDotProduct.apply(q, k, v)
    out = torch.einsum('...nd,...n->...nd', out, D_inv)

    if is_half:
        out = out.half()

    return out

# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v):
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k.cumsum(dim=-2))
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context = context.cumsum(dim=-3)
    out = torch.einsum('...nde,...nd,...n->...ne', context, q, D_inv)
    return out

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, feature_redraw_interval = 0, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, amp_enabled = False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0)) # Make sure this is persistent

        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows = self.nb_features, nb_columns = dim_heads, scaling = ortho_scaling, qr_uniform_q = qr_uniform_q)
        projection_matrix = self.create_projection()
        self.register_buffer('projection_matrix', projection_matrix)

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = partial(causal_linear_attention, amp_enabled = amp_enabled)
            except ImportError:
#                 print('unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def forward(self, q, k, v):
        device = q.device

        # It's time to redraw the projection matrix
        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            self.projection_matrix = self.create_projection(device = device).type_as(q)
            self.calls_since_last_redraw = torch.tensor(0)
        # Keep track of how many forward passes we do before we redraw again
        else:
            self.calls_since_last_redraw += 1

        if self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))
        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out

# classes

class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(1e-3))
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.g

class PreScaleNorm(nn.Module):
    def __init__(self, dim, fn, eps=1e-5):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.ones(1))
        self.eps = eps

    def forward(self, x, **kwargs):
        n = torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        x = x / n * self.g
        return self.fn(x, **kwargs)

class PreLayerNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim = -1):
        super().__init__()
        self.dim = along_dim
        self.chunks = chunks
        self.fn = fn

    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        chunks = x.chunk(self.chunks, dim = self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in chunks], dim = self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, nn.GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, causal = False, heads = 8, local_heads = 0, local_window_size = 256, nb_features = None, feature_redraw_interval = 1000, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, dropout = 0., amp_enabled = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = dim // heads
        self.fast_attention = FastAttention(dim_head, nb_features, feature_redraw_interval, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, amp_enabled = amp_enabled)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context = None, mask = None, context_mask = None):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)
        context = default(context, x)
        context_mask = default(context_mask, mask)

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                k.masked_fill_(~global_mask, 0)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)

class Performer(nn.Module):
    def __init__(self, dim, depth, heads, local_attn_heads = 0, local_window_size = 256, causal = False, ff_mult = 4, nb_features = None, reversible = False, ff_chunks = 1, generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, use_scalenorm = False, use_rezero = False, ff_glu = False, ff_dropout = 0., attn_dropout = 0., cross_attend = False, amp_enabled = False):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):
            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, causal = causal, heads = heads, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, dropout = attn_dropout, amp_enabled = amp_enabled)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

            if not cross_attend:
                continue

            layers.append(nn.ModuleList([
                wrapper_fn(SelfAttention(dim, heads = heads, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, qr_uniform_q = qr_uniform_q, dropout = attn_dropout)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)

class PerformerLM(nn.Module):
    def __init__(self, *, num_tokens, max_seq_len, dim, depth, heads, local_attn_heads = 0, local_window_size = 256, causal = False, ff_mult = 4, nb_features = None, reversible = False, ff_chunks = 1, ff_glu = False, emb_dropout = 0., ff_dropout = 0., attn_dropout = 0., generalized_attention = False, kernel_fn = nn.ReLU(), qr_uniform_q = False, use_scalenorm = False, use_rezero = False, cross_attend = False, amp_enabled = False):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(emb_dropout)

        nn.init.normal_(self.token_emb.weight, std = 0.02)
        nn.init.normal_(self.pos_emb.weight, std = 0.02)

        self.performer = Performer(dim, depth, heads, local_attn_heads, local_window_size, causal, ff_mult, nb_features, reversible, ff_chunks, generalized_attention, kernel_fn, qr_uniform_q, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, cross_attend, amp_enabled)
        self.norm = nn.LayerNorm(dim)

    def fix_projection_matrices_(self):
        fast_attentions = find_modules(self, FastAttention)
        device = get_module_device(self)
        for fast_attention in fast_attentions:
            fast_attention.feature_redraw_interval = None

    def forward(self, x, return_encodings = False, **kwargs):
        b, n, device = *x.shape, x.device
        # token and positional embeddings
        x = self.token_emb(x)
        x += self.pos_emb(torch.arange(n, device = device))
        x = self.dropout(x)

        # performer layers
        x = self.performer(x, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if return_encodings:
            return x

        return x @ self.token_emb.weight.t()