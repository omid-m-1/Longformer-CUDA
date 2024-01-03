inline void export_kernel(py::module &m) { 
    m.def("lformerMM",[](py::capsule& input1, py::capsule& input2, py::capsule& output1, py::capsule& dilation, bool no_dilation, int Window, int WindowUpper, bool transposeT1, bool transposeT2, bool GPU){
        array4d_t<float> input1_array = capsule_to_array4d<float>(input1);
        array4d_t<float> input2_array = capsule_to_array4d<float>(input2);
        array4d_t<float> output1_array = capsule_to_array4d<float>(output1);
        array1d_t<int> dilation_array = capsule_to_array1d<int>(dilation);
    return lformerMM(input1_array, input2_array, output1_array, dilation_array, no_dilation, Window, WindowUpper, transposeT1, transposeT2, GPU);
    }
  );
}