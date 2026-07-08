#pragma once

// Per-model compile-time constants for the native runtime. One model per
// binary: select the multilingual profile with -DNEMOTRON_PROFILE_ML (CMake:
// -DNEMOTRON_PROFILE=ml); the default build serves the English model.
//
// These mirror runtime/model_profile.py (the Python export-side source of
// truth) and are cross-checked at load time against the artifact manifests
// (finalize bucket manifest CONTRACT, steady-batch MANIFEST, session-bundle
// meta), so a profile/artifact mismatch fails closed at startup.

#ifdef NEMOTRON_PROFILE_ML

static constexpr int BLANK = 13087;
static constexpr int MAX_SYMBOLS = 10;
static constexpr int SHIFT = 32;
static constexpr int PRE = 9;
static constexpr int DROP = 2;
static constexpr int RIGHT_CONTEXT = 3;
static constexpr int FINAL_PADDING_FRAMES = 128;  // (RIGHT_CONTEXT + 1) * SHIFT
static constexpr int ATT_CONTEXT_LEFT = 56;
static constexpr int ATT_CONTEXT_RIGHT = 3;
static constexpr const char* MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b";
// Language-ID prompt conditioning: a [1, NUM_PROMPTS] one-hot per session is
// applied to every encoder output through prompt_apply.ts (a small MLP)
// before RNNT decode. See model_profile.py and PromptStreamingMixin in NeMo.
static constexpr bool MODEL_PROMPTED = true;
static constexpr int NUM_PROMPTS = 128;

#else

static constexpr int BLANK = 1024;
static constexpr int MAX_SYMBOLS = 10;
static constexpr int SHIFT = 16;
static constexpr int PRE = 9;
static constexpr int DROP = 2;
static constexpr int RIGHT_CONTEXT = 1;
static constexpr int FINAL_PADDING_FRAMES = 32;  // (RIGHT_CONTEXT + 1) * SHIFT
static constexpr int ATT_CONTEXT_LEFT = 70;
static constexpr int ATT_CONTEXT_RIGHT = 1;
static constexpr const char* MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b";
static constexpr bool MODEL_PROMPTED = false;
static constexpr int NUM_PROMPTS = 0;

#endif
