"""
tests/twilio_integration_test.py
──────────────────────────────────
Full Twilio integration test.

Tests the complete call flow end-to-end:
  1. Verifies Twilio credentials are working
  2. Makes an outbound test call from Twilio to your webhook
  3. Watches the call flow through the system
  4. Reports what happened

Prerequisites:
  - Your .env has valid TWILIO_* credentials
  - Your webhook service is running (python -m services.webhook.main)
  - Your server is reachable from the internet (ngrok or AWS public IP)

Usage:
  python tests/twilio_integration_test.py --webhook-url https://your-ngrok-url.ngrok.io
  python tests/twilio_integration_test.py --webhook-url http://<AWS-IP>
"""
import asyncio
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from twilio.rest import Client as TwilioClient
from config import get_settings
from logger import get_logger

log = get_logger("twilio_test")
settings = get_settings()


def test_twilio_credentials() -> bool:
    """Verifies Twilio credentials are valid."""
    print("\n[1/4] Verifying Twilio credentials...")
    try:
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        account = client.api.accounts(settings.twilio_account_sid).fetch()
        print(f"      ✅ Account: {account.friendly_name} (status: {account.status})")
        return True
    except Exception as e:
        print(f"      ❌ Failed: {e}")
        print(f"         Check TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN in .env")
        return False


def test_webhook_reachable(webhook_url: str) -> bool:
    """Checks your webhook is publicly reachable."""
    print(f"\n[2/4] Checking webhook is reachable at {webhook_url}...")
    try:
        import httpx as _httpx
        with _httpx.Client(timeout=10) as client:
            r = client.get(f"{webhook_url}/health")
            data = r.json()
            print(f"      ✅ Webhook reachable: status={data.get('status')}, "
                  f"slots={data.get('available_slots')}")
            return True
    except Exception as e:
        print(f"      ❌ Not reachable: {e}")
        print(f"         Make sure your webhook is running and accessible from the internet.")
        print(f"         On AWS: ensure port 80 is open in your Security Group.")
        return False


def configure_twilio_webhook(webhook_url: str) -> bool:
    """Sets the Twilio phone number webhook to your server."""
    print(f"\n[3/4] Configuring Twilio webhook URL...")
    try:
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)

        # Find the phone number
        numbers = client.incoming_phone_numbers.list(
            phone_number=settings.twilio_phone_number
        )

        if not numbers:
            print(f"      ❌ Phone number {settings.twilio_phone_number} not found in account")
            return False

        phone_sid = numbers[0].sid

        # Update webhook URLs
        client.incoming_phone_numbers(phone_sid).update(
            voice_url=f"{webhook_url}/twilio/incoming",
            voice_method="POST",
            status_callback=f"{webhook_url}/twilio/status",
            status_callback_method="POST",
        )

        print(f"      ✅ Webhook configured:")
        print(f"         Voice URL:       {webhook_url}/twilio/incoming")
        print(f"         Status callback: {webhook_url}/twilio/status")
        return True

    except Exception as e:
        print(f"      ❌ Failed to configure webhook: {e}")
        return False


def make_test_call(webhook_url: str) -> bool:
    """
    Makes a real outbound test call using Twilio.
    The call will be answered by your AI agent.
    """
    print(f"\n[4/4] Making test call to {settings.twilio_phone_number}...")

    if not settings.twilio_phone_number or settings.twilio_phone_number == "+1xxxxxxxxxx":
        print("      ⚠️  No Twilio phone number configured.")
        print("         Set TWILIO_PHONE_NUMBER in .env to your Twilio trial number.")
        print("         You can still test using the simulate endpoint:")
        print(f"         curl -X POST '{webhook_url}/calls/simulate?room_name=test-1'")
        return False

    try:
        client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)

        # On Twilio free trial, you can only call verified numbers
        # The call_to number must be verified in your Twilio console
        call_to = settings.twilio_phone_number  # call your own number

        call = client.calls.create(
            to=call_to,
            from_=settings.twilio_phone_number,
            url=f"{webhook_url}/twilio/incoming",
            status_callback=f"{webhook_url}/twilio/status",
            status_callback_method="POST",
        )

        print(f"      ✅ Call initiated!")
        print(f"         Call SID:  {call.sid}")
        print(f"         Status:    {call.status}")
        print(f"         Watch it:  https://console.twilio.com/us1/monitor/logs/calls/{call.sid}")
        print()
        print(f"      ⏳ Waiting 10s then checking call status...")

        time.sleep(10)

        call = client.calls(call.sid).fetch()
        print(f"      📞 Call status after 10s: {call.status}")

        return True

    except Exception as e:
        print(f"      ❌ Call failed: {e}")
        print()
        print("      Common issues with Twilio free trial:")
        print("        - Can only call verified numbers (verify at console.twilio.com)")
        print("        - Must have non-zero account balance")
        print("        - Use /calls/simulate for testing without real calls")
        return False


def run_simulation_test(webhook_url: str):
    """Alternative test using the simulate endpoint — no Twilio needed."""
    print(f"\n  Running simulation test (no Twilio phone call needed)...")
    try:
        import httpx as _httpx
        with _httpx.Client(timeout=15) as client:
            r = client.post(f"{webhook_url}/calls/simulate?room_name=integration-test-1")
            r.raise_for_status()
            data = r.json()

        print(f"  ✅ Simulation successful!")
        print(f"     Call ID:    {data['call_id']}")
        print(f"     Room:       {data['room_name']}")
        print(f"     Join URL:   {data.get('join_url', 'N/A')}")
        print()
        print(f"  👉 To talk to the AI agent:")
        print(f"     Open this URL in your browser:")
        print(f"     {data.get('join_url', 'N/A')}")
        print()
        print(f"  👉 Or use lk CLI:")
        print(f"     lk room join --url {settings.livekit_url} --token {data['caller_token']}")

    except Exception as e:
        print(f"  ❌ Simulation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test Twilio + Voice AI Agent integration")
    parser.add_argument(
        "--webhook-url",
        default=f"http://localhost:{settings.webhook_port}",
        help="Public URL of your webhook service (e.g. http://<AWS-IP> or ngrok URL)",
    )
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="Only run simulation test (no real Twilio call)",
    )
    args = parser.parse_args()

    webhook_url = args.webhook_url.rstrip("/")

    print("=" * 60)
    print("  Voice AI Agent — Twilio Integration Test")
    print("=" * 60)

    if args.simulate_only:
        ok = test_webhook_reachable(webhook_url)
        if ok:
            run_simulation_test(webhook_url)
        return

    # Full integration test
    ok = test_twilio_credentials()
    if not ok:
        run_simulation_test(webhook_url)
        return

    ok = test_webhook_reachable(webhook_url)
    if not ok:
        return

    ok = configure_twilio_webhook(webhook_url)
    if not ok:
        print("\n  Falling back to simulation test...")
        run_simulation_test(webhook_url)
        return

    make_test_call(webhook_url)

    print("\n" + "=" * 60)
    print("  Integration test complete.")
    print("  Check Grafana for metrics: http://<your-ip>/grafana")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
