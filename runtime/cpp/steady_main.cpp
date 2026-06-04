// 1.2b (T1-hardened) — C++ STREAMING steady pipeline, the verified Python streaming loop ported + paired-review fixes:
//  - per-session StreamState (reset-able; no cross-utterance contamination)   [Codex#5]
//  - metadata assertions vs the bundle (no silent geometry/drop mismatch)      [Codex#4]
//  - FULL-chunk-only steady loop: process complete SHIFT chunks; the remainder (<SHIFT) is the FINALIZE path's job
//    (server-faithful: server.py's steady only fires on full shift_frames) -> fixes the partial/short-chunk crash    [Codex#1, Opus#2]
//  - range checks on encoder enc_len                                            [Codex#6]
// T1 (token-exact vs NeMo streaming). Encoder byte-exactness (T2a) = the next torch.export/dynamic step.
#include "lib/runtime_io/jit_load.h"

#include <torch/script.h>
#include <cstdio>
#include <vector>
#include <stdexcept>

static constexpr int BLANK = 1024, MAX_SYMBOLS = 10, SHIFT = 16, PRE = 9, DROP = 2;

// Per-session streaming state — reset between utterances/sessions (the 1.0 state-ownership model, in miniature).
struct StreamState {
  torch::Tensor clc, clt, clcl;   // encoder cache
  torch::Tensor g, h, c;          // decoder pred-output + LSTM state
  torch::Tensor ring;             // last PRE mel frames
  int emitted = 0;
  std::vector<int64_t> hyp;
  void reset(torch::jit::Module& init, torch::jit::Module& sb) {
    clc = sb.attr("clc0").toTensor().clone(); clt = sb.attr("clt0").toTensor().clone(); clcl = sb.attr("clcl0").toTensor().clone();
    g = init.attr("sos_g").toTensor().clone(); h = init.attr("init_h").toTensor().clone(); c = init.attr("init_c").toTensor().clone();
    ring = torch::Tensor(); emitted = 0; hyp.clear();
  }
};

// One steady chunk: select first/steady geometry, run encoder, thread cache, decode (state-carry). Mutates st.
static void steady_chunk(StreamState& st, const torch::Tensor& new_mel,
                         torch::jit::Module& enc_first, torch::jit::Module& enc_steady,
                         torch::jit::Module& joint, torch::jit::Module& predict) {
  auto CU = torch::kCUDA;
  torch::Tensor chunk; torch::jit::Module* mod;
  if (st.emitted == 0) { chunk = new_mel; mod = &enc_first; }
  else { chunk = torch::cat({st.ring, new_mel}, 2); mod = &enc_steady; }
  auto L = torch::full({1}, chunk.size(2), torch::dtype(torch::kLong).device(CU));
  auto out = mod->forward({chunk, L, st.clc, st.clt, st.clcl}).toTuple();
  auto eo = out->elements()[0].toTensor();
  int To = out->elements()[1].toTensor().item<int64_t>();
  if (To < 0 || To > eo.size(2))                                  // range check [Codex#6]
    throw std::runtime_error("enc_len To=" + std::to_string(To) + " out of range for enc_out frames=" + std::to_string(eo.size(2)));
  st.clc = out->elements()[2].toTensor(); st.clt = out->elements()[3].toTensor(); st.clcl = out->elements()[4].toTensor();

  auto f = eo.transpose(1, 2).contiguous();
  for (int t = 0; t < To; ++t) {
    auto f_t = f.slice(1, t, t + 1);
    for (int n = 0; n < MAX_SYMBOLS; ++n) {
      int64_t k = joint.forward({f_t, st.g}).toTensor().reshape({-1}).argmax().item<int64_t>();
      if (k == BLANK) break;
      st.hyp.push_back(k);
      auto y = torch::full({1, 1}, k, torch::dtype(torch::kLong).device(CU));
      auto o = predict.forward({y, st.h, st.c}).toTuple();
      st.g = o->elements()[0].toTensor(); st.h = o->elements()[1].toTensor(); st.c = o->elements()[2].toTensor();
    }
  }
  auto cum = st.ring.defined() ? torch::cat({st.ring, new_mel}, 2) : new_mel;
  st.ring = cum.slice(2, std::max<int64_t>(0, cum.size(2) - PRE), cum.size(2));
  st.emitted += new_mel.size(2);
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "../artifacts";
  torch::NoGradGuard ng; auto CU = torch::kCUDA;
  auto enc_first  = load_jit_serialized(dir + "/enc_first.ts");  enc_first.to(CU);  enc_first.eval();
  auto enc_steady = load_jit_serialized(dir + "/enc_steady.ts"); enc_steady.to(CU); enc_steady.eval();
  auto joint   = load_jit_serialized(dir + "/joint_step.ts");    joint.to(CU);   joint.eval();
  auto predict = load_jit_serialized(dir + "/predict_step.ts");  predict.to(CU); predict.eval();
  auto init    = load_jit_serialized(dir + "/cpp_bundle.ts");    init.to(CU);
  auto sb      = load_jit_serialized(dir + "/stream_bundle.ts"); sb.to(CU);

  // metadata assertion vs compiled constants [Codex#4]
  auto meta = sb.attr("meta").toTensor().to(torch::kCPU);
  int m_shift=meta[0].item<int64_t>(), m_pre=meta[1].item<int64_t>(), m_drop=meta[2].item<int64_t>(),
      m_blank=meta[3].item<int64_t>(), m_msym=meta[4].item<int64_t>();
  if (m_shift!=SHIFT||m_pre!=PRE||m_drop!=DROP||m_blank!=BLANK||m_msym!=MAX_SYMBOLS) {
    std::printf("METADATA MISMATCH bundle(shift=%d pre=%d drop=%d blank=%d msym=%d) vs compiled(%d %d %d %d %d)\n",
                m_shift,m_pre,m_drop,m_blank,m_msym,SHIFT,PRE,DROP,BLANK,MAX_SYMBOLS);
    return 2;
  }

  auto mel  = sb.attr("mel").toTensor();
  auto gold = sb.attr("gold").toTensor().to(torch::kCPU);
  int Tm = mel.size(2);

  StreamState st; st.reset(init, sb);
  int pos = 0, nchunks = 0;
  for (; pos + SHIFT <= Tm; pos += SHIFT, ++nchunks)              // FULL chunks only; remainder -> finalize
    steady_chunk(st, mel.slice(2, pos, pos + SHIFT), enc_first, enc_steady, joint, predict);
  int remainder = Tm - pos;                                       // deferred to the finalize path (1.3)

  bool ok = ((int)st.hyp.size() == gold.size(0));
  for (int i = 0; ok && i < (int)st.hyp.size(); ++i) ok = (st.hyp[i] == gold[i].item<int64_t>());
  std::printf("C++ STREAMING: chunks=%d remainder=%d (->finalize) -> %zu tokens vs gold %ld -> %s (T1 token-exact)\n",
              nchunks, remainder, st.hyp.size(), (long)gold.size(0), ok ? "PASS" : "FAIL");

  // robustness: reset + re-run on a truncated NON-multiple-of-16 mel (must not crash; processes floor chunks)
  int Tt = Tm - 9;                                                // 311 = 19*16 + 7 -> partial remainder
  st.reset(init, sb);
  int n2 = 0;
  try {
    for (int p = 0; p + SHIFT <= Tt; p += SHIFT, ++n2)
      steady_chunk(st, mel.slice(2, p, p + SHIFT), enc_first, enc_steady, joint, predict);
    std::printf("[robustness] Tm=%d (non-mult-16): %d full chunks, remainder=%d, %zu tokens, NO CRASH -> PASS\n",
                Tt, n2, Tt - n2*SHIFT, st.hyp.size());
  } catch (const std::exception& e) { std::printf("[robustness] CRASH: %s\n", e.what()); return 1; }
  return ok ? 0 : 1;
}
