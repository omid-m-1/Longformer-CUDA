import torch as th
from . import gp_apis

class lformerMM_impl(th.autograd.Function):
    @staticmethod
    def _out_size(input1: th.Tensor, input2: th.Tensor, window: int, dilation: th.Tensor,
                       is_diagonal: bool = False, is_transposed: bool = False,
                       autoregressive: bool = False):
        assert len(input1.shape) == 4
        assert len(input1.shape) == len(input2.shape)
        assert input1.shape[:3] == input2.shape[:3]
        assert len(dilation.shape) == 1
        assert dilation.shape[0] == input1.shape[2]
        dim1_0 = input1.shape[0]
        dim1_1 = input1.shape[1]
        dim1_2 = input1.shape[2]
        hidden_dim = input2.shape[3]
        w_upper = 0 if autoregressive else window
        diagonal = w_upper + window + 1
        if is_diagonal:
            assert input1.shape[3] == diagonal
            dim1_3 = hidden_dim
        else:
            assert not is_transposed
            assert input1.shape[3] == hidden_dim
            dim1_3 = diagonal
        if hidden_dim == diagonal:
            print('Error: the hidden dimension shouldn\'t match number of diagonals')
            assert False
        return dim1_0, dim1_1, dim1_2, dim1_3

    @staticmethod
    def _prepare_tensors(t):
        assert t.is_contiguous()
        t_stride = list(t.stride())
        t_size = list(t.size())
        if t_size[0] == 1 and t_stride[0] == t_stride[1]:
            t_stride[0] = t_size[1] * t_size[2] * t_size[3]
            t = t.as_strided(size=t_size, stride=t_stride)
        return t

    @staticmethod
    def forward(ctx, input1, input2, window, dilation, is_diagonal, padding, autoregressive, coalesced):
        device0 = input1.device.type
        input1 = lformerMM_impl._prepare_tensors(input1)
        input2 = lformerMM_impl._prepare_tensors(input2)
        if isinstance(dilation, int):
            dilation = input1.new_full(size=(input1.shape[2],), fill_value=dilation, dtype=th.int, requires_grad=False, device=device0)
        dim1_0, dim1_1, dim1_2, dim1_3 = lformerMM_impl._out_size(input1, input2, window, dilation, is_diagonal, autoregressive=autoregressive)
        params = th.tensor([window, 0 if autoregressive else window, padding, 0, coalesced])
        mode3 = 1 if (input1.shape[3] == input2.shape[3]) else 0
        res = gp_apis.gp_lformerMM(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, dilation, params, device0, mode3)
        ctx.backward_cache = (input1, input2, dilation) #must be implemented
        ctx.window = window
        ctx.is_diagonal = is_diagonal
        ctx.autoregressive = autoregressive
        ctx.device0 = device0
        ctx.coalesced = coalesced
        return res

    @staticmethod
    def backward(ctx, dZ):
        input1, input2, dilation = ctx.backward_cache
        window = ctx.window
        is_diagonal = ctx.is_diagonal
        autoregressive = ctx.autoregressive
        device0 = ctx.device0
        coalesced = ctx.coalesced
        if not dZ.is_contiguous():
            dZ = dZ.contiguous()
        dZ = lformerMM_impl._prepare_tensors(dZ)
        params = th.tensor([window, 0 if autoregressive else window, 0, 0, coalesced])
        is_diagonal = not is_diagonal
        dim1_0, dim1_1, dim1_2, dim1_3 = lformerMM_impl._out_size(dZ, input2, window, dilation, is_diagonal, autoregressive=autoregressive)
        mode3 = 1 if (dZ.shape[3] == input2.shape[3]) else 0
        grd1 = gp_apis.gp_lformerMM(dZ, input2, dim1_0, dim1_1, dim1_2, dim1_3, dilation, params, device0, mode3)
        params = th.tensor([window, 0 if autoregressive else window, 0, 1, coalesced])
        mode3 = 1 if (dZ.shape[3] == input1.shape[3]) else 0
        if is_diagonal:
            dim1_0, dim1_1, dim1_2, dim1_3 = lformerMM_impl._out_size(dZ, input1, window, dilation, True, True, autoregressive=autoregressive)
            grd2 = gp_apis.gp_lformerMM(dZ, input1, dim1_0, dim1_1, dim1_2, dim1_3, dilation, params, device0, mode3)
        else:
            dim1_0, dim1_1, dim1_2, dim1_3 = lformerMM_impl._out_size(input1, dZ, window, dilation, True, True, autoregressive=autoregressive)
            grd2 = gp_apis.gp_lformerMM(input1, dZ, dim1_0, dim1_1, dim1_2, dim1_3, dilation, params, device0, mode3)
        return grd1, grd2, None, None, None, None, None, None

def lformerMM(input1, input2, window, dilation, is_diagonal = False, padding = 0, autoregressive = False, coalesced = 0):
    return lformerMM_impl.apply(input1, input2, window, dilation, is_diagonal, padding, autoregressive, coalesced)
