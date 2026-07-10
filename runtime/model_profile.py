#!/usr/bin/env python3
"""Single source of truth for per-model export constants.

Every export/validation script derives its model id, attention context, chunk
geometry, and decode constants from a ModelProfile instead of local literals.
The active profile is selected with NEMOTRON_EXPORT_PROFILE (default: "en",
which preserves the original English-only pipeline byte-for-byte).

Profiles:
  en  nvidia/nemotron-speech-streaming-en-0.6b   att [70,1], vocab 1024
  ml  nvidia/nemotron-3.5-asr-streaming-0.6b     att [56,3], vocab 13087,
      language-ID prompt conditioning (post-encoder MLP, see apply_prompt)

The "ml" model applies its language prompt to the ENCODER OUTPUT
(PromptStreamingMixin._apply_prompt_to_encoded): enc_out (B,D,T) is
transposed, concatenated per-frame with a num_prompts one-hot, passed through
model.prompt_kernel (Linear -> ReLU -> Linear), and transposed back. The
cache-aware encoder graphs themselves are language-independent, so exported
encoder EPs carry no prompt input; the MLP is exported separately
(prompt_apply.ts) and the reference decode paths call apply_prompt() after
every cache_aware_stream_step.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model_id: str
    att_context: tuple[int, int]  # (left, right) in encoder frames
    shift: int  # mel frames consumed per steady chunk (streaming_cfg.shift_size[1])
    pre: int  # pre-encode cache mel frames (streaming_cfg.pre_encode_cache_size[1])
    drop: int  # streaming_cfg.drop_extra_pre_encoded
    mels: int  # preprocessor feature dim
    blank: int  # RNNT blank id == tokenizer vocab size
    max_symbols: int
    prompted: bool  # language-ID prompt conditioning (post-encoder MLP)
    num_prompts: int = 0  # one-hot width of the prompt vector
    default_target_lang: str = ""
    default_prompt_index: int = -1  # prompt_dictionary[default_target_lang]

    @property
    def right_context(self) -> int:
        return self.att_context[1]

    @property
    def final_padding_frames(self) -> int:
        return (self.right_context + 1) * self.shift

    @property
    def drop0_T(self) -> range:
        start = self.final_padding_frames + 2
        return range(start, start + self.shift)

    @property
    def drop2_T(self) -> range:
        start = self.final_padding_frames + 2 + self.pre
        return range(start, start + self.shift)

    @property
    def fixtures_dirname(self) -> str:
        """Per-profile golden-fixture directory ("fixtures" stays the en legacy name)."""
        return "fixtures" if self.name == "en" else f"fixtures_{self.name}"


PROFILES: dict[str, ModelProfile] = {
    "en": ModelProfile(
        name="en",
        model_id="nvidia/nemotron-speech-streaming-en-0.6b",
        att_context=(70, 1),
        shift=16,
        pre=9,
        drop=2,
        mels=128,
        blank=1024,
        max_symbols=10,
        prompted=False,
    ),
    "ml": ModelProfile(
        name="ml",
        model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        att_context=(56, 3),
        shift=32,
        pre=9,
        drop=2,
        mels=128,
        blank=13087,
        max_symbols=10,
        prompted=True,
        num_prompts=128,
        default_target_lang="auto",
        default_prompt_index=101,
    ),
}


def get_profile() -> ModelProfile:
    name = os.environ.get("NEMOTRON_EXPORT_PROFILE", "en")
    try:
        return PROFILES[name]
    except KeyError:
        raise SystemExit(
            f"unknown NEMOTRON_EXPORT_PROFILE={name!r}; choose from {sorted(PROFILES)}"
        )


def load_profile_model(profile: ModelProfile | None = None, target_lang: str | None = None):
    """Load the profile's checkpoint, pin its att context, validate geometry.

    For prompted profiles, sets the inference prompt (default_target_lang
    unless overridden) so reference decode paths condition like production.
    """
    import nemo.collections.asr as nemo_asr

    profile = profile or get_profile()
    model = (
        nemo_asr.models.ASRModel.from_pretrained(profile.model_id, map_location="cpu")
        .cuda()
        .eval()
    )
    try:
        model.preprocessor.featurizer.dither = 0.0
    except Exception:
        pass
    model.encoder.set_default_att_context_size(list(profile.att_context))
    model.encoder.setup_streaming_params()
    if profile.prompted:
        model.set_inference_prompt(target_lang or profile.default_target_lang)
    validate_model_against_profile(model, profile)
    return model


def validate_model_against_profile(model, profile: ModelProfile) -> None:
    """Fail closed if the checkpoint disagrees with the profile constants."""
    scfg = model.encoder.streaming_cfg
    to_int = lambda v: int(v[1]) if isinstance(v, (list, tuple)) else int(v)
    checks = {
        "shift": (to_int(scfg.shift_size), profile.shift),
        "pre": (to_int(scfg.pre_encode_cache_size), profile.pre),
        "drop": (int(scfg.drop_extra_pre_encoded), profile.drop),
        "att_left": (int(scfg.last_channel_cache_size), profile.att_context[0]),
        "mels": (int(model.cfg.preprocessor.features), profile.mels),
        "blank": (int(model.joint.num_classes_with_blank) - 1, profile.blank),
    }
    if profile.prompted:
        checks["num_prompts"] = (int(model.num_prompts), profile.num_prompts)
        table = prompt_dictionary(model)
        checks["default_prompt_index"] = (
            int(table.get(profile.default_target_lang, -1)),
            profile.default_prompt_index,
        )
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise RuntimeError(
            f"profile {profile.name!r} does not match checkpoint {profile.model_id!r}: "
            + ", ".join(f"{k}: model={a} profile={b}" for k, (a, b) in mismatches.items())
        )


def apply_prompt(model, enc_out):
    """Post-encoder language conditioning; identity for non-prompted models.

    Mirrors PromptStreamingMixin._apply_prompt_to_encoded so reference decode
    matches production streaming exactly. Uses getattr so the same call sites
    work on NeMo builds that predate the prompt mixin.
    """
    hook = getattr(model, "_apply_prompt_to_encoded", None)
    if hook is None:
        return enc_out
    return hook(enc_out)


def prompt_dictionary(model) -> dict[str, int]:
    """locale -> prompt one-hot index table (empty for non-prompted models)."""
    table = model.cfg.model_defaults.get("prompt_dictionary", {}) if hasattr(model, "cfg") else {}
    return {str(k): int(v) for k, v in dict(table).items()}
