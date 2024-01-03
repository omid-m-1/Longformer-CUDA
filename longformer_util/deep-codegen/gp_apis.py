import torch as th
import torch.utils.dlpack
from . import graphpy as gpk
def gp_lformerMM(input1, input2, dim1_0, dim1_1, dim1_2, dim1_3, dilation, no_dilation, window, windowupper, transposeT1, device0):
    if (input2.shape[3] == dim1_3) :
        input2 = input2.permute(0, 3, 2, 1).contiguous()
        transposeT2 = True
    else:
        transposeT2 = False
    input1_dl = th.utils.dlpack.to_dlpack(input1)
    input2_dl = th.utils.dlpack.to_dlpack(input2)
    dilation_dl = th.utils.dlpack.to_dlpack(dilation)
    res1 = th.zeros(dim1_0, dim1_1, dim1_2, dim1_3, device = device0)
    res_dl1 = th.utils.dlpack.to_dlpack(res1)
    GPU = True if device0 == 'cuda' else False
    gpk.lformerMM(input1_dl, input2_dl, res_dl1, dilation_dl, no_dilation, window, windowupper, transposeT1, transposeT2, GPU)
    if (input2.shape[3] == dim1_3) : input2 = input2.permute(0, 3, 2, 1).contiguous()
    return res1
