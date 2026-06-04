// 1.2b-pre — full C++ ASR pipeline: audio -> preproc -> encoder -> greedy decode -> tokens, self-checked BYTE-EXACT vs
// gold. Proves preproc + encoder + decode COMPOSE byte-exact in C++ on real speech. (Full non-streaming encoder; the
// streaming cache-aware chunk loop is 1.2b proper.) Build: CMakeLists.txt (manual-link libtorch, no nvcc).
#include "lib/runtime_io/jit_load.h"

#include <torch/script.h>
#include <cstdio>
#include <vector>

static constexpr int BLANK = 1024, MAX_SYMBOLS = 10;

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng;
  auto preproc = load_jit_serialized(dir + "/preproc.ts");        preproc.to(torch::kCUDA); preproc.eval();
  auto encoder = load_jit_serialized(dir + "/encoder_full.ts");   encoder.to(torch::kCUDA); encoder.eval();
  auto joint   = load_jit_serialized(dir + "/joint_step.ts");     joint.to(torch::kCUDA);   joint.eval();
  auto predict = load_jit_serialized(dir + "/predict_step.ts");   predict.to(torch::kCUDA); predict.eval();
  auto bundle  = load_jit_serialized(dir + "/pipeline_bundle.ts");bundle.to(torch::kCUDA);
  auto init    = load_jit_serialized(dir + "/cpp_bundle.ts");     init.to(torch::kCUDA);   // reuse sos_g/init_h/init_c

  auto audio = bundle.attr("audio").toTensor();                       // [1, N]
  auto alen  = bundle.attr("alen").toTensor();                        // [1]
  auto gold  = bundle.attr("gold").toTensor().to(torch::kCPU);

  // audio -> mel -> encoder
  auto pp = preproc.forward({audio, alen}).toTuple();
  auto proc = pp->elements()[0].toTensor(); auto proc_len = pp->elements()[1].toTensor();
  auto eo = encoder.forward({proc, proc_len}).toTuple();
  auto enc = eo->elements()[0].toTensor();                            // [1,1024,T]
  int T = eo->elements()[1].toTensor().item<int64_t>();
  std::printf("pipeline: mel %s -> enc %s (T=%d)\n",
              std::to_string(proc.size(2)).c_str(), std::to_string(enc.size(2)).c_str(), T);

  // greedy decode (the verified loop)
  auto g = init.attr("sos_g").toTensor(), h = init.attr("init_h").toTensor(), c = init.attr("init_c").toTensor();
  auto f = enc.transpose(1, 2).contiguous();
  std::vector<int64_t> hyp;
  for (int t = 0; t < T; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      int64_t k = joint.forward({f_t, g}).toTensor().reshape({-1}).argmax().item<int64_t>();
      if (k == BLANK) break;
      hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(torch::kCUDA));
      auto out = predict.forward({y, h, c}).toTuple();
      g = out->elements()[0].toTensor(); h = out->elements()[1].toTensor(); c = out->elements()[2].toTensor();
    }
  }

  bool ok = ((int)hyp.size() == gold.size(0));
  for (int i = 0; ok && i < (int)hyp.size(); ++i) ok = (hyp[i] == gold[i].item<int64_t>());
  std::printf("FULL C++ PIPELINE: %zu tokens vs gold %ld -> %s\n",
              hyp.size(), (long)gold.size(0), ok ? "BYTE-EXACT PASS" : "FAIL");
  if (!ok) { std::printf("  got :"); for (auto k : hyp) std::printf(" %ld", (long)k);
             std::printf("\n  gold:"); for (int i=0;i<gold.size(0);++i) std::printf(" %ld",(long)gold[i].item<int64_t>()); std::printf("\n"); }
  return ok ? 0 : 1;
}
