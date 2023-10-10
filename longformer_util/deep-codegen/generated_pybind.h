inline void export_kernel(py::module &m) { 
    m.def("lformerMM",[](py::capsule& input1, py::capsule& input2, py::capsule& output1, py::capsule& dilation, py::capsule& params, bool GPU){
        array4d_t<float> input1_array = capsule_to_array4d<float>(input1);
        array4d_t<float> input2_array = capsule_to_array4d<float>(input2);
        array4d_t<float> output1_array = capsule_to_array4d<float>(output1);
        array1d_t<int> dilation_array = capsule_to_array1d<int>(dilation);
        array1d_t<int> params_array = capsule_to_array1d<int>(params);
    return lformerMM(input1_array, input2_array, output1_array, dilation_array, params_array, GPU);
    }
  );
}
