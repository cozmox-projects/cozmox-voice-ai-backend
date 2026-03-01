"""
services/webhook/livekit_manager.py
─────────────────────────────────────
Creates LiveKit rooms and generates access tokens.
Used by the webhook service when a new call comes in.
"""

import asyncio

from livekit.api import AccessToken, CreateRoomRequest, LiveKitAPI, VideoGrants

from config import get_settings
from logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class LiveKitManager:
    """Manages LiveKit room creation and token generation."""

    async def create_room(self, room_name: str) -> str:
        """
        Creates a LiveKit room for the call.
        Returns the WebSocket URL for Twilio to connect to.
        """
        try:
            async with LiveKitAPI(
                url=settings.livekit_url.replace("ws://", "http://").replace(
                    "wss://", "https://"
                ),
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            ) as lk_api:
                await lk_api.room.create_room(
                    CreateRoomRequest(
                        name=room_name,
                        empty_timeout=300,  # close room after 5 min of inactivity
                        max_participants=10,
                    )
                )
                log.info("livekit_room_created", room=room_name)

        except Exception as e:
            # Room may already exist — that's fine
            log.debug("livekit_room_create_info", room=room_name, info=str(e))

        return room_name

    def generate_caller_token(self, room_name: str, caller_identity: str) -> str:
        """
        Generates a LiveKit access token for the Twilio caller to join the room.
        Twilio uses this token when it connects to LiveKit via WebSocket.
        """
        token = (
            AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(caller_identity)
            .with_name("Caller")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,  # caller publishes their voice
                    can_subscribe=True,  # caller receives agent voice
                )
            )
            .to_jwt()
        )
        return token

    def get_livekit_ws_url(self, room_name: str, token: str) -> str:
        """Returns the WebSocket URL that Twilio should connect to."""
        lk_ws = settings.livekit_url
        return f"{lk_ws}?token={token}"


# Singleton
_livekit_manager = None


def get_livekit_manager() -> LiveKitManager:
    global _livekit_manager
    if _livekit_manager is None:
        _livekit_manager = LiveKitManager()
    return _livekit_manager
