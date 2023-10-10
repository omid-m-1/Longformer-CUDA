import tensorflow as tf
import gp_apis

def lformerMM(input1, input2, dilation, params, device0):
    @tf.custom_gradient
    def _lambda(X1, X2):
        return lformerMM_real(X1, X2, dilation, params, device0)
    return _lambda(input1, input2)

def lformerMM_real(input1, input2, dilation, params, device0):
    out = gp_apis.gp_lformerMM(input1, input2, dilation, params, device0)
    def grad(dZ1, dZ2):
        return gp_apis.gp_lformerMM(dZ1, dZ2, dilation, params, device0)
    return out, grad

