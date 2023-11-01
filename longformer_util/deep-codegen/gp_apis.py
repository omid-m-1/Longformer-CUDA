import torch as th
import torch.utils.dlpack
from . import graphpy as gpk
def gp_lformerMM(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, dilation, params, device0, mode3):
    if params[-1]:
        input1 = input1.transpose(2,3)
        input1.contiguous()
        if mode3:
            input2 = input2.transpose(2,3)
            input2.contiguous()
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    dilation_dl = th.utils.dlpack.to_dlpack(dilation)
    params_dl = th.utils.dlpack.to_dlpack(params)
    res1 = th.zeros(dim1_0, dim1_1, dim1_2, dim1_3, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    GPU = True if device0 == 'cuda' else False
    gpk.lformerMM(input1_dl, input2_dl, res_dl1, dilation_dl, params_dl, GPU)
    return res1
