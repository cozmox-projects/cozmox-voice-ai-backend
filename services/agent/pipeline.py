"""
services/agent/pipeline.py
───────────────────────────
The core AI pipeline for a single call.

Flow:
  LiveKit Audio In
      → Deepgram STT (streaming)
      → Turn Detector (end-of-speech)
      → Knowledge Base Retriever (RAG context)
      → Azure OpenAI (streaming response)
      → ElevenLabs TTS (streaming audio)
      → LiveKit Audio Out

Barge-in is handled by BargeInController running alongside this pipeline.
One instance of this class is created per call.
"""

import asyncio
import time
from typing import Any, Optional

import aiohttp

# Deepgram v3 — use asyncwebsocket (asynclive deprecated since 3.4.0)
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

# ElevenLabs via LiveKit plugin — handles WebSocket streaming + MP3/PCM decode internally
from livekit.plugins.elevenlabs import TTS as ElevenLabsTTS
from openai import AsyncAzureOpenAI

from config import get_settings
from logger import get_logger
from services.agent.barge_in import BargeInController
from services.agent.prompts import RAG_CONTEXT_TEMPLATE, SYSTEM_PROMPT
from services.agent.turn_detector import TurnDetector
from services.knowledge.retriever import KnowledgeRetriever
from services.metrics.collector import (
    llm_errors_total,
    stt_errors_total,
    tts_errors_total,
)
from services.metrics.latency_tracker import LatencyTracker
from services.resilience.circuit_breaker import llm_circuit_breaker

log = get_logger(__name__)
settings = get_settings()


class VoiceAIPipeline:
    """
    Manages the complete STT → LLM → TTS pipeline for one call.

    Args:
        call_id:    Unique ID for this call (e.g. "call-twilio-CA123")
        room_name:  LiveKit room name this agent is connected to
        send_audio: Async callable that sends audio bytes back to LiveKit
    """

    def __init__(self, call_id: str, room_name: str, send_audio_callback):
        self.call_id = call_id
        self.room_name = room_name
        self._send_audio = send_audio_callback

        # Sub-components
        self.barge_in = BargeInController(call_id=call_id)
        self.turn_detector = TurnDetector(
            call_id=call_id, on_turn_end=self._on_turn_end
        )
        self.retriever = KnowledgeRetriever()
        self.latency_tracker = LatencyTracker(call_id=call_id)

        # Azure OpenAI client
        self._llm_client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )

        # ElevenLabs TTS via livekit-plugins-elevenlabs.
        # We pass our own aiohttp.ClientSession because this plugin normally
        # expects to run inside a LiveKit agent worker job context which sets
        # up a shared session automatically. Since we run it standalone, we
        # create and manage the session ourselves.
        self._tts_session: Optional[aiohttp.ClientSession] = None
        self._tts = ElevenLabsTTS(
            voice_id=settings.elevenlabs_voice_id,
            model="eleven_turbo_v2_5",
            api_key=settings.elevenlabs_api_key,
            http_session=None,  # will be set in start()
        )

        # Conversation history (keep last 10 turns to control token count)
        self._conversation_history = []
        self._max_history_turns = 10

        # State
        self._is_running = False
        self._current_tts_task: Optional[asyncio.Task] = None

        log.info("pipeline_created", call_id=call_id, room=room_name)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self):
        """Start the pipeline. Called after connecting to LiveKit room."""
        import time

        self._pipeline_start_time = time.perf_counter()
        self._is_running = True

        # Create the aiohttp session now that we're inside an async context
        self._tts_session = aiohttp.ClientSession()
        self._tts._session = self._tts_session

        log.info("pipeline_started", call_id=self.call_id)

        # Greet the caller
        await self._speak("Hello! Thank you for calling. How can I help you today?")

    async def stop(self):
        """Gracefully shut down the pipeline."""
        self._is_running = False
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
        if self._tts_session and not self._tts_session.closed:
            await self._tts_session.close()

    # ── Deepgram connection ───────────────────────────────────────────────────

    async def create_deepgram_connection(self):
        """
        Creates a live Deepgram connection for this call.
        Returns the connection object that audio bytes should be sent to.
        """
        dg_client = DeepgramClient(settings.deepgram_api_key)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            interim_results=True,
            utterance_end_ms="1000",
            vad_events=True,
            # LiveKit AudioStream always delivers int16 PCM frames regardless
            # of the source encoding. Do NOT use "mulaw" here — that caused
            # Deepgram to receive PCM bytes it interpreted as mulaw, producing
            # zero transcripts.
            encoding="linear16",
            sample_rate=8000,
            channels=1,
        )

        # asyncwebsocket is the current API (asynclive deprecated in 3.4+)
        connection = dg_client.listen.asyncwebsocket.v("1")

        # ── Deepgram event handlers ───────────────────────────────────────

        async def on_transcript(self_dg, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                if not sentence:
                    return
                self.turn_detector.on_transcript_update(sentence)
                if result.is_final:
                    log.debug("stt_final", call_id=self.call_id, text=sentence)
            except Exception as e:
                stt_errors_total.inc()
                log.error("stt_error", call_id=self.call_id, error=str(e))

        async def on_speech_started(self_dg, speech_started, **kwargs):
            self.turn_detector.on_speech_start()
            # Guard: ignore speech events in the first second after pipeline start.
            # The agent's own TTS greeting audio can loop back through the room
            # and trigger a false barge-in before the caller has said anything.
            import time

            if (
                hasattr(self, "_pipeline_start_time")
                and (time.perf_counter() - self._pipeline_start_time) < 1.0
            ):
                return
            if self.barge_in.on_speech_detected():
                await self._cancel_current_tts()

        async def on_utterance_end(self_dg, utterance_end, **kwargs):
            self.turn_detector.on_speech_end()
            self.latency_tracker.speech_ended()

        async def on_error(self_dg, error, **kwargs):
            stt_errors_total.inc()
            log.error("deepgram_error", call_id=self.call_id, error=str(error))

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.SpeechStarted, on_speech_started)
        connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)
        connection.on(LiveTranscriptionEvents.Error, on_error)

        await connection.start(options)
        log.info("deepgram_connection_open", call_id=self.call_id)

        # Keepalive: Deepgram closes idle connections after ~12s without audio.
        # During agent TTS playback, no caller audio is sent. This task pings
        # Deepgram every 8s to keep the connection alive for the call's duration.
        async def _keepalive():
            try:
                while True:
                    await asyncio.sleep(8)
                    await connection.keep_alive()
            except Exception:
                pass  # connection closed, task ends naturally

        asyncio.ensure_future(_keepalive())

        return connection

    # ── Turn end → LLM ────────────────────────────────────────────────────────

    async def _on_turn_end(self, transcript: str):
        """
        Called by TurnDetector when the caller finishes speaking.
        Runs the full LLM → TTS pipeline.
        """
        if not transcript.strip():
            return

        log.info("processing_turn", call_id=self.call_id, transcript=transcript)
        self.latency_tracker.stt_completed(transcript)

        # 1. Retrieve relevant knowledge base context
        kb_context = await self.retriever.retrieve(transcript)

        # 2. Build messages for LLM
        messages = self._build_messages(transcript, kb_context)

        # 3. Stream LLM response → TTS
        await self._llm_to_tts(messages)

    def _build_messages(self, user_text: str, kb_context: Optional[str]) -> list:
        """Builds the OpenAI messages array with system prompt + history + user input."""
        system_content = SYSTEM_PROMPT
        if kb_context:
            system_content += RAG_CONTEXT_TEMPLATE.format(context=kb_context)
        messages = [{"role": "system", "content": system_content}]

        # Add conversation history (last N turns)
        messages.extend(self._conversation_history[-self._max_history_turns :])

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    # ── LLM ───────────────────────────────────────────────────────────────────

    async def _llm_to_tts(self, messages: list):
        """
        Streams LLM response token by token.
        As tokens arrive, builds sentences and pipes them to TTS.
        This is the key to low latency — we don't wait for the full response.
        """

        async def _call_llm():
            return await self._llm_client.chat.completions.create(
                model=settings.azure_openai_deployment,
                messages=messages,
                stream=True,
                max_tokens=150,
                temperature=0.7,
            )

        try:
            stream = await llm_circuit_breaker.call(_call_llm)

            # If circuit breaker returned fallback string
            if isinstance(stream, str):
                await self._speak(stream)
                return

            full_response = ""
            sentence_buffer = ""
            first_token = True

            async for chunk in stream:
                if not self._is_running:
                    break
                delta = chunk.choices[0].delta if chunk.choices else None
                token = getattr(delta, "content", None) if delta else None
                if not token:
                    continue
                if first_token:
                    self.latency_tracker.llm_first_token()
                    first_token = False
                full_response += token
                sentence_buffer += token

                # Send to TTS as soon as we have a complete sentence
                # This starts audio playback before the full response is done
                if any(sentence_buffer.endswith(p) for p in [".", "!", "?", ","]):
                    if len(sentence_buffer.strip()) > 10:
                        self._current_tts_task = asyncio.create_task(
                            self._speak(sentence_buffer.strip())
                        )
                        await self._current_tts_task
                        sentence_buffer = ""

            # Speak any remaining text
            if sentence_buffer.strip() and self._is_running:
                await self._speak(sentence_buffer.strip())

            # Save to conversation history
            if full_response:
                self._conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            llm_errors_total.inc()
            log.error("llm_error", call_id=self.call_id, error=str(e))
            await self._speak("I'm sorry, I had a problem. Could you repeat that?")

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def _speak(self, text: str):
        """
        Converts text to speech via the livekit-plugins-elevenlabs plugin.

        The plugin connects to ElevenLabs via WebSocket (/multi-stream-input),
        receives base64-encoded MP3 chunks in JSON messages, decodes them via
        PyAV (the 'av' package), and yields raw int16 PCM frames at 22050 Hz.

        We iterate the ChunkedStream and pipe each PCM AudioFrame directly
        into our LiveKit AudioSource. No format detection, no pydub, no ffmpeg.
        Works on free tier because MP3 is the native format the plugin expects.
        """
        if not text.strip():
            return
        self.barge_in.on_tts_started()
        first_frame = True
        try:
            async with self._tts.synthesize(text) as stream:
                async for audio_event in stream:
                    if self.barge_in._interrupt_event.is_set():
                        break

                    if first_frame:
                        self.latency_tracker.tts_first_audio()
                        self.latency_tracker.report()
                        first_frame = False

                    # audio_event.frame is an rtc.AudioFrame with:
                    #   sample_rate=22050, num_channels=1, data=int16 PCM bytes
                    await self._send_audio(bytes(audio_event.frame.data))

        except asyncio.CancelledError:
            log.info("tts_task_cancelled", call_id=self.call_id)
        except Exception as e:
            tts_errors_total.inc()
            log.error("tts_error", call_id=self.call_id, error=str(e))
        finally:
            self.barge_in.on_tts_stopped()

    async def _cancel_current_tts(self):
        """Cancels TTS playback immediately (for barge-in)."""
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
            try:
                await self._current_tts_task
            except asyncio.CancelledError:
                pass
        log.debug("tts_cancelled", call_id=self.call_id)
