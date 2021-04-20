#include <torch/extension.h>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

/*
std::vector<at::Tensor> conv2d_cosSinAlpha_forward(
        torch::Tensor input,    // shape: (b, n_in, h, w)
        torch::Tensor weights,  // shape: (n_out, n_in, K)
        torch::Tensor real_kernels, 
        torch::Tensor imag_kernels,  // shape: (n_out, n_in, K)
        torch::Tensor cos_sin_alpha) {
    torch::
}

std::vector<at::Tensor> conv2d_cosSinKAlpha_forward(
        torch::Tensor input,    // shape: (b, n_in, h, w)
        torch::Tensor weights,  // shape: (n_out, n_in, K)
        torch::Tensor real_kernels,
        torch::Tensor imag_kernels,  // shape: (n_out, n_in, K)
        torch::Tensor cos_sin_kalpha) {
    
}
*/

torch::Tensor cos_sin_ka(const torch::Tensor &cos_a, const torch::Tensor &sin_a, 
                          const torch::Tensor &cos_km1_a, const torch::Tensor &sin_km1_a) {
    // cos(ka) = cos(a) cos( (k-1)a ) - sin(a) sin( (k-1)a )
    auto cos_ka = cos_a*cos_km1_a;
    cos_ka.addcmul_(sin_a, sin_km1_a, -1);

    // sin(ka) = cos(a) sin( (k-1)a ) + sin(a) cos( (k-1)a )
    auto sin_ka = cos_a*sin_km1_a;
    sin_ka.addcmul_(sin_a, cos_km1_a);
    return torch::stack({cos_ka, sin_ka});
}

/**
    Computes the matrix:
        [[cos(α), cos(2α), ..., cos(kα)],
         [sin(α), sin(2α), ..., sin(kα)]]

        Args:
            cos_sin_a: The tensor [cos(α), sin(α)] of shape [2, b, n_out, h, w]

            k: The max k

        Returns: The tensor cos_sin_ka of shape [2, k, b, n_out, h, w], where:
                    cos_sin_ka[0] = [cos(α), cos(2α), ..., cos(kα)]
                    cos_sin_ka[1] = [sin(α), sin(2α), ..., sin(kα)]

torch::ArrayRef<torch::TensorList> cos_sin_ka_stack(const torch::Tensor &cos_a, const torch::Tensor &sin_a, int k){
    torch::TensorList out[k];
    out[0] = torch::TensorList({cos_a, sin_a});
    for(int i=1; i<k; i++) {
        out[i] = cos_sin_ka(cos_a, sin_a, out[i-1][0], out[i-1][1]);
    }
    return torch::ArrayRef<torch::TensorList>(out, k);
}
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //m.def("forward", &conv2d_forward, "steerable_kbase conv2d forward");
  //m.def("cos_sin_ka_stack", &cos_sin_ka_stack, "cos_sin_ka_stack");
  m.def("cos_sin_ka", &cos_sin_ka, "cos_sin_ka");
}
