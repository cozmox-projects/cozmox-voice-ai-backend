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
from typing import Optional, AsyncGenerator

from openai import AsyncAzureOpenAI
import httpx

from config import get_settings
from logger import get_logger
from services.agent.prompts import SYSTEM_PROMPT, RAG_CONTEXT_TEMPLATE
from services.agent.barge_in import BargeInController
from services.agent.turn_detector import TurnDetector
from services.knowledge.retriever import KnowledgeRetriever
from services.metrics.latency_tracker import LatencyTracker
from services.metrics.collector import (
    stt_errors_total,
    llm_errors_total,
    tts_errors_total,
)
from services.resilience.circuit_breaker import llm_circuit_breaker
from services.resilience.retry_policy import retry_ai_call

import deepgram
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

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

    def __init__(
        self,
        call_id: str,
        room_name: str,
        send_audio_callback,
    ):
        self.call_id = call_id
        self.room_name = room_name
        self._send_audio = send_audio_callback

        # Sub-components
        self.barge_in = BargeInController(call_id=call_id)
        self.turn_detector = TurnDetector(
            call_id=call_id,
            on_turn_end=self._on_turn_end,
        )
        self.retriever = KnowledgeRetriever()
        self.latency_tracker = LatencyTracker(call_id=call_id)

        # Azure OpenAI client
        self._llm_client = AsyncAzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
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
        self._is_running = True
        log.info("pipeline_started", call_id=self.call_id)

        # Greet the caller
        await self._speak(
            "Hello! Thank you for calling Acme Corp. How can I help you today?"
        )

    async def stop(self):
        """Gracefully shut down the pipeline."""
        self._is_running = False
        if self._current_tts_task and not self._current_tts_task.done():
            self._current_tts_task.cancel()
        log.info("pipeline_stopped", call_id=self.call_id)

    # ── Audio input (called by LiveKit audio receiver) ────────────────────────

    async def process_audio_chunk(self, audio_bytes: bytes):
        """
        Entry point for incoming audio from the caller.
        Audio chunks are fed into Deepgram's streaming STT.
        """
        if not self._is_running:
            return
        # In real Pipecat integration this is handled by the transport layer.
        # This method exists for direct/test usage.
        pass

    # ── Deepgram STT integration ──────────────────────────────────────────────

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
            interim_results=True,       # stream partial results for lower latency
            utterance_end_ms=1000,      # used alongside our turn detector
            vad_events=True,            # VAD events for barge-in detection
            encoding="mulaw",           # Twilio sends μ-law audio
            sample_rate=8000,           # standard telephone audio
            channels=1,
        )

        connection = dg_client.listen.asynclive.v("1")

        # ── Deepgram event handlers ───────────────────────────────────────

        async def on_transcript(self_dg, result, **kwargs):
            try:
                sentence = result.channel.alternatives[0].transcript
                is_final = result.is_final

                if not sentence:
                    return

                self.turn_detector.on_transcript_update(sentence)

                if is_final:
                    log.debug(
                        "stt_final_transcript",
                        call_id=self.call_id,
                        text=sentence,
                    )

            except Exception as e:
                stt_errors_total.inc()
                log.error("stt_transcript_error", call_id=self.call_id, error=str(e))

        async def on_speech_started(self_dg, speech_started, **kwargs):
            self.turn_detector.on_speech_start()
            was_barge_in = self.barge_in.on_speech_detected()
            if was_barge_in:
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
        messages.extend(self._conversation_history[-self._max_history_turns:])

        # Add current user message
        messages.append({"role": "user", "content": user_text})

        return messages

    # ── LLM (Azure OpenAI) ────────────────────────────────────────────────────

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
                max_tokens=150,      # keep responses short for voice
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

                token = chunk.choices[0].delta.content if chunk.choices else None
                if token is None:
                    continue

                if first_token:
                    self.latency_tracker.llm_first_token()
                    first_token = False

                full_response += token
                sentence_buffer += token

                # Send to TTS as soon as we have a complete sentence
                # This starts audio playback before the full response is done
                if any(sentence_buffer.endswith(p) for p in [".", "!", "?", ","]):
                    if len(sentence_buffer.strip()) > 10:  # skip tiny fragments
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
            await self._speak(
                "I'm sorry, I encountered an issue. Could you repeat that?"
            )

    # ── ElevenLabs TTS ────────────────────────────────────────────────────────

    async def _speak(self, text: str):
        """
        Converts text to speech via ElevenLabs and sends audio to LiveKit.
        Uses streaming — first audio chunk arrives in ~150ms.
        """
        if not text.strip():
            return

        self.barge_in.on_tts_started()

        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}/stream"

            headers = {
                "xi-api-key": settings.elevenlabs_api_key,
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "model_id": "eleven_turbo_v2_5",   # fastest model
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "speed": 1.0,
                },
                "output_format": "ulaw_8000",        # μ-law 8kHz for Twilio
            }

            first_chunk = True

            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_bytes(chunk_size=1024):
                        # Check for barge-in before sending each chunk
                        if self.barge_in._interrupt_event.is_set():
                            log.info("tts_cancelled_barge_in", call_id=self.call_id)
                            break

                        if first_chunk:
                            self.latency_tracker.tts_first_audio()
                            self.latency_tracker.report()  # record complete E2E latency
                            first_chunk = False

                        # Send audio back to caller via LiveKit
                        await self._send_audio(chunk)

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
