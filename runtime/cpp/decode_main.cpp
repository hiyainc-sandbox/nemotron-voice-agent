// 1.2a — C++ RNNT greedy decode (the verified reference, translated to libtorch C++). Loads the exported
// joint_step.ts + predict_step.ts + a self-contained cpp_bundle.ts (init constants + the real_1 fixture enc/gold as
// named buffers), runs the greedy loop, and self-checks BYTE-EXACT vs the gold y_sequence. Proves the decode is
// byte-exact in C++ (B1b). Build: CMakeLists.txt (manual-link libtorch, no nvcc — same as the 0.1b microbench).
#include <torch/script.h>
#include "lib/runtime_io/jit_load.h"
#include <cstdio>
#include <vector>

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;

// Decode encoder frames [t0,t1) of f:[1,T,1024], carrying (pred output g, LSTM state h,c). RESUMABLE — the
// streaming-continuation contract (matches ref_decode.py). Appends token ids to hyp; mutates g,h,c.
static void decode_range(torch::jit::Module& joint, torch::jit::Module& predict, const torch::Tensor& f,
                         int t0, int t1, torch::Tensor& g, torch::Tensor& h, torch::Tensor& c,
                         std::vector<int64_t>& hyp) {
  auto dev = f.device();
  for (int t = t0; t < t1; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      auto logits = joint.forward({f_t, g}).toTensor();
      int64_t k = logits.reshape({-1}).argmax().item<int64_t>();
      if (k == BLANK) break;
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(dev));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor(); h = out->elements()[1].toTensor(); c = out->elements()[2].toTensor();
    }
  }
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng;
  auto joint   = load_jit_serialized(dir + "/joint_step.ts");   joint.to(torch::kCUDA);   joint.eval();
  auto predict = load_jit_serialized(dir + "/predict_step.ts"); predict.to(torch::kCUDA); predict.eval();
  auto bundle  = load_jit_serialized(dir + "/cpp_bundle.ts");   bundle.to(torch::kCUDA);

  auto g    = bundle.attr("sos_g").toTensor();          // [1,1,640]
  auto h    = bundle.attr("init_h").toTensor();         // [2,1,640]
  auto c    = bundle.attr("init_c").toTensor();
  auto enc  = bundle.attr("enc").toTensor();            // [1,1024,T]
  int  T    = bundle.attr("enc_len").toTensor().item<int64_t>();
  auto gold = bundle.attr("gold").toTensor().to(torch::kCPU);

  auto f = enc.transpose(1, 2).contiguous();            // [1,T,1024]
  auto g0 = g.clone(), h0 = h.clone(), c0 = c.clone();

  // full-utterance decode
  std::vector<int64_t> hyp;
  decode_range(joint, predict, f, 0, T, g, h, c, hyp);

  // streaming: decode in 2 chunks carrying state -> must equal full (the state-carry contract)
  std::vector<int64_t> hstream;
  { auto gg = g0.clone(), hh = h0.clone(), cc = c0.clone();
    decode_range(joint, predict, f, 0, T / 2, gg, hh, cc, hstream);
    decode_range(joint, predict, f, T / 2, T, gg, hh, cc, hstream); }

  bool ok = ((int)hyp.size() == gold.size(0));
  for (int i = 0; ok && i < (int)hyp.size(); ++i) ok = (hyp[i] == gold[i].item<int64_t>());
  bool stream_ok = (hstream == hyp);
  std::printf("C++ decode: %zu tokens vs gold %ld -> %s | streaming(2-chunk carry)==full -> %s\n",
              hyp.size(), (long)gold.size(0), ok ? "BYTE-EXACT PASS" : "FAIL", stream_ok ? "PASS" : "FAIL");
  if (!ok) {
    std::printf("  got :"); for (auto k : hyp) std::printf(" %ld", (long)k); std::printf("\n  gold:");
    for (int i = 0; i < gold.size(0); ++i) std::printf(" %ld", (long)gold[i].item<int64_t>()); std::printf("\n");
  }
  return (ok && stream_ok) ? 0 : 1;
}
