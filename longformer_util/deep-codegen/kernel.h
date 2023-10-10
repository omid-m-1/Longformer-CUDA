#pragma once
#include "csr.h"
#include "op.h"

void lformerMM(array4d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, array1d_t<int>& dilation, array1d_t<int>& params, bool GPU);
