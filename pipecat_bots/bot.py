#!/usr/bin/env python3
#
# Low-latency voice agent built with Pipecat, using NVIDIA open models.
#
#   Turn detection : Pipecat Smart Turn v3 (on-device ONNX, the 1.2.x default)
#   STT            : Nemotron streaming STT (WebSocket) — endpoint from NVIDIA_ASR_URL
#   LLM            : Nemotron-3 Nano (OpenAI-compatible /v1) — endpoint from NVIDIA_LLM_URL
#   TTS            : on-device Pocket TTS (Kyutai) — endpoint from POCKET_TTS_URL
#   Transport      : SmallWebRTCTransport + the default prebuilt UI
#
# The ASR and LLM endpoints are REQUIRED env vars — point them at the model servers
# you run locally (RTX 5090 / DGX Spark unified container) or anywhere reachable.
#
# Run:
#   uv run pipecat_bots/bot.py            # SmallWebRTC, open the URL it prints
#   uv run pipecat_bots/bot.py -t webrtc  # same, explicit
#
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams

from nemotron_llm import VLLMOpenAILLMService
from nvidia_stt import NVidiaWebSocketSTTService
from pocket_tts import PocketTTSService

load_dotenv(override=True)


def _require_env(name: str, example: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required env var {name}. Point it at your model server, e.g.\n"
            f"  {name}={example}"
        )
    return value


# REQUIRED — endpoints for the ASR and LLM model servers.
NVIDIA_ASR_URL = _require_env("NVIDIA_ASR_URL", "ws://localhost:8080")
NVIDIA_LLM_URL = _require_env("NVIDIA_LLM_URL", "http://localhost:8000/v1")
NVIDIA_LLM_MODEL = os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-nano")
NVIDIA_LLM_API_KEY = os.getenv("NVIDIA_LLM_API_KEY", "EMPTY")  # local servers ignore this
# Reasoning OFF by default for low-latency voice (only enable if the server runs a
# reasoning parser, else chain-of-thought is spoken aloud).
NEMOTRON_ENABLE_THINKING = os.getenv("NEMOTRON_ENABLE_THINKING", "false").lower() == "true"

# Pocket TTS runs on-device; defaults to a local server (see README to start it).
POCKET_TTS_URL = os.getenv("POCKET_TTS_URL", "http://127.0.0.1:8001")
POCKET_TTS_VOICE = os.getenv("POCKET_TTS_VOICE", "alba")

# SmallWebRTC is the default/intended transport. In pipecat 1.2.x, VAD and turn
# detection are configured on the LLM user aggregator (see run_bot), not here.
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Nemotron low-latency voice agent")
    logger.info(f"  STT: Nemotron streaming @ {NVIDIA_ASR_URL}")
    logger.info(f"  LLM: {NVIDIA_LLM_MODEL} @ {NVIDIA_LLM_URL} (thinking={NEMOTRON_ENABLE_THINKING})")
    logger.info(f"  TTS: Pocket TTS @ {POCKET_TTS_URL} (voice={POCKET_TTS_VOICE})")

    stt = NVidiaWebSocketSTTService(url=NVIDIA_ASR_URL, sample_rate=16000, strip_interim_prefix=True)

    llm = VLLMOpenAILLMService(
        api_key=NVIDIA_LLM_API_KEY,
        base_url=NVIDIA_LLM_URL,
        settings=VLLMOpenAILLMService.Settings(
            model=NVIDIA_LLM_MODEL,
            extra={"extra_body": {"chat_template_kwargs": {"enable_thinking": NEMOTRON_ENABLE_THINKING}}},
        ),
    )

    tts = PocketTTSService(base_url=POCKET_TTS_URL, voice=POCKET_TTS_VOICE)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, friendly voice assistant. Your responses are "
                "spoken aloud, so keep them short and conversational. Use plain text "
                "only: no markdown, emojis, bullet points, or special characters. Get "
                "to the point quickly."
            ),
        },
        {"role": "user", "content": "Greet the user in one short sentence and ask how you can help."},
    ]

    context = LLMContext(messages)
    # VAD + Smart Turn v3 live on the user aggregator in pipecat 1.2.x. Passing only
    # vad_analyzer uses the defaults: VAD at stop_secs=0.2 and the bundled on-device
    # Smart Turn v3 model. VAD frames are broadcast upstream, so the STT service sees
    # VADUserStoppedSpeakingFrame and finalizes on it.
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )

    # Observability: per-service TTFB/processing metrics + end-to-end voice-to-voice latency.
    metrics_logger = MetricsLogObserver()
    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency_measured(observer, latency_seconds):
        logger.info(f"🗣️→🤖 voice-to-voice latency: {latency_seconds * 1000:.0f} ms")

    # RTVI is enabled by default: PipelineTask auto-inserts the RTVIProcessor and an
    # RTVIObserver that forwards metrics/transcripts/events to the client UI.
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[metrics_logger, latency_observer],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("RTVI client ready")
        await task.queue_frames([LLMRunFrame()])  # kick off the spoken greeting

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Entry point for the Pipecat runner."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
