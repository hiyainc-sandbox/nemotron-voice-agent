// 1.2b-wire-C++ (closes action-D BLOCKER #1/#6): load the AOTI steady-encoder package from a NATIVE C++ runtime via
// torch::inductor::AOTIModelPackageLoader, feed the same t2a_io fixture, and (a) compare byte-for-byte vs the eager
// reference — expected to REPRODUCE the Python aoti_load_package result (enc_out 1.99e-5, cache_t 1.66e-2), proving the
// C++ path is numerically identical to Python; (b) exercise the C++-only seams the review flagged: explicit CUDA stream,
// device index, output ownership/aliasing across calls, and non-contiguous inputs.
//
// Build: in the container, manual-link libtorch (CMakeLists aoti_encoder target). Run: ./aoti_encoder <artifacts_dir>
#include "lib/runtime_io/jit_load.h"

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <vector>
#include <string>

using torch::inductor::AOTIModelPackageLoader;

static const char* NAMES[5] = {"enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"};

// byte-for-byte + max-abs-diff compare of two output vectors against the eager reference
static bool compare(const std::vector<at::Tensor>& got, const std::vector<at::Tensor>& ref, const char* tag) {
  bool allok = true; double maxd = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    auto a = ref[i].to(torch::kCUDA), b = got[i].to(torch::kCUDA);
    bool shape_ok = a.sizes() == b.sizes();
    bool eq = shape_ok && at::equal(a, b);
    double d = (shape_ok && a.numel() > 0) ? (a.to(torch::kFloat) - b.to(torch::kFloat)).abs().max().item<double>() : NAN;
    allok = allok && eq;
    if (!(d != d)) maxd = std::max(maxd, d);
    printf("    %-12s byte-equal=%d max_abs_diff=%.3e\n", NAMES[i], (int)eq, d);
  }
  printf("  [%s] %s (maxdiff %.3e)\n", tag, allok ? "BYTE-EXACT" : "NOT byte-exact", maxd);
  return allok;
}

int main(int argc, char** argv) {
  torch::NoGradGuard ng;
  std::string art = (argc > 1) ? argv[1] : "../artifacts";
  std::string pkg = art + "/enc_steady_aoti.pt2";

  // fixture: scripted bundle of the 5 inputs + 5 eager-reference outputs (built by the driver)
  auto io = load_jit_serialized(art + "/t2a_io_bundle.ts");
  std::vector<at::Tensor> inputs = {
      io.attr("chunk").toTensor().to(torch::kCUDA), io.attr("L").toTensor().to(torch::kCUDA),
      io.attr("clc").toTensor().to(torch::kCUDA),   io.attr("clt").toTensor().to(torch::kCUDA),
      io.attr("clcl").toTensor().to(torch::kCUDA)};
  std::vector<at::Tensor> ref, pyaoti;
  for (int i = 0; i < 5; ++i) ref.push_back(io.attr(std::string("out") + std::to_string(i)).toTensor().to(torch::kCUDA));
  for (int i = 0; i < 5; ++i) pyaoti.push_back(io.attr(std::string("aoti") + std::to_string(i)).toTensor().to(torch::kCUDA));

  printf("=== C++ AOTIModelPackageLoader: %s ===\n", pkg.c_str());

  // (1) default load + run (default stream, device auto).
  AOTIModelPackageLoader loader(pkg, "model", /*run_single_threaded=*/false, /*num_runners=*/1, /*device_index=*/-1);
  printf("loaded; call_spec entries=%zu\n", loader.get_call_spec().size());
  auto out_default = loader.run(inputs);
  // THE headline check: C++ output == Python aoti_load_package output, BYTE-FOR-BYTE (closes "same .so -> same numerics").
  printf("(1a) C++ run() vs PYTHON aoti_load_package output (must be byte-exact):\n");
  bool ok1 = compare(out_default, pyaoti, "cpp-vs-python");
  // context: both sit at the SAME 1.66e-2 vs the eager reference (the known compiled-encoder T1 residual).
  printf("(1b) C++ run() vs EAGER reference (context — expect the same 1.66e-2 as Python):\n");
  compare(out_default, ref, "cpp-vs-eager");

  // (2) explicit CUDA stream seam — run on a non-default stream; numerics must match the default-stream run
  auto stream = c10::cuda::getStreamFromPool(/*isHighPriority=*/false, /*device=*/0);
  auto out_stream = loader.run(inputs, (void*)stream.stream());
  c10::cuda::device_synchronize();
  printf("(2) explicit non-default CUDA stream:\n");
  bool ok2 = compare(out_stream, out_default, "explicit-stream-vs-default");

  // (3) explicit device index seam — construct a loader pinned to device 0; must match
  AOTIModelPackageLoader loader0(pkg, "model", false, 1, /*device_index=*/0);
  auto out_dev0 = loader0.run(inputs);
  printf("(3) explicit device_index=0 (vs Python aoti output):\n");
  bool ok3 = compare(out_dev0, pyaoti, "device0-vs-python");

  // (4) output ownership / aliasing seam — keep call-1 outputs, run again, ensure call-1 results are NOT mutated
  std::vector<at::Tensor> keep;
  for (auto& t : out_default) keep.push_back(t.clone());
  auto out_again = loader.run(inputs);  // second call into the same runner
  bool alias_ok = true;
  for (size_t i = 0; i < keep.size(); ++i)
    if (!at::equal(keep[i].to(torch::kCUDA), out_default[i].to(torch::kCUDA))) alias_ok = false;
  printf("(4) output ownership: call-1 outputs unmutated after a 2nd run = %d\n", (int)alias_ok);

  // (5) non-contiguous input seam — feed a non-contiguous 'chunk' (slice of a padded tensor); model should still match
  auto wide = torch::empty({inputs[0].size(0), inputs[0].size(1), inputs[0].size(2) + 4}, inputs[0].options());
  wide.narrow(2, 0, inputs[0].size(2)).copy_(inputs[0]);
  auto chunk_nc = wide.narrow(2, 0, inputs[0].size(2));  // non-contiguous view
  std::vector<at::Tensor> in_nc = {chunk_nc, inputs[1], inputs[2], inputs[3], inputs[4]};
  // SEAM CHARACTERIZATION (not pass/fail): run() does NOT defensively make inputs contiguous. A non-contiguous input
  // silently produces WRONG output (no throw) -> the C++ runtime MUST .contiguous() all inputs before run().
  printf("(5) non-contiguous chunk seam (is_contiguous=%d) — expect DIVERGENCE (documents the contiguity requirement):\n", (int)chunk_nc.is_contiguous());
  bool nc_diverges = false;
  try {
    auto out_nc = loader.run(in_nc);
    bool nc_match = compare(out_nc, out_default, "noncontig-vs-default");
    nc_diverges = !nc_match;
    // and confirm .contiguous() FIXES it
    std::vector<at::Tensor> in_fixed = {chunk_nc.contiguous(), inputs[1], inputs[2], inputs[3], inputs[4]};
    auto out_fixed = loader.run(in_fixed);
    printf("    after .contiguous() (must match default):\n");
    compare(out_fixed, out_default, "contiguous-fix");
  } catch (const std::exception& e) { printf("    threw: %s\n", e.what()); }

  bool all = ok1 && ok2 && ok3 && alias_ok;
  printf("=== ACTION D CLOSE: C++==Python(byte-exact)=%d | stream-invariant=%d | device0==Python=%d | no-alias=%d | noncontig-diverges(req .contiguous)=%d ===\n",
         (int)ok1, (int)ok2, (int)ok3, (int)alias_ok, (int)nc_diverges);
  printf("=== %s ===\n", all ? "C++ AOTI path VALIDATED: byte-for-byte identical to Python aoti_load_package; seams characterized (inputs MUST be contiguous)"
                             : "C++ AOTI path MISMATCH vs Python — investigate");
  return all ? 0 : 1;
}
