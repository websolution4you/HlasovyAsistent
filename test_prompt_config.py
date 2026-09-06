import asyncio
import json
import os
import unittest
from unittest.mock import patch

from starlette.requests import Request

import main
from call_context import verify_call_context


class PromptConfigTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "VOICE_CALL_CONTEXT_SECRET": "test-call-context-secret",
                "ELEVENLABS_INIT_WEBHOOK_SECRET": "test-webhook-secret",
            },
            clear=False,
        )
        self.environment.start()
        main.CALL_CONTEXT.clear()
        main.CONVERSATION_CONTEXT.clear()

    def tearDown(self):
        main.CALL_CONTEXT.clear()
        main.CONVERSATION_CONTEXT.clear()
        self.environment.stop()

    @staticmethod
    def request(body: dict, secret: str = "") -> Request:
        headers = [(b"content-type", b"application/json")]
        if secret:
            headers.append((b"x-elevenlabs-webhook-secret", secret.encode("ascii")))
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/prompt-config",
                "query_string": b"tenant=ntc",
                "headers": headers,
            }
        )
        request._body = json.dumps(body).encode("utf-8")
        return request

    def test_trusted_sip_webhook_returns_signed_provider_neutral_context(self):
        body = {
            "caller_id": "+421900000001",
            "called_number": "+421200000000",
            "call_sid": "telnyx-provider-call-id",
            "call_id": "sip-dialog-id",
            "conversation_id": "conv_test",
        }
        with patch.object(main, "find_user_name_and_id_by_phone", return_value=("Ján Novák", "user-1")):
            result = asyncio.run(
                main.prompt_config(self.request(body, "test-webhook-secret"))
            )

        variables = result["dynamic_variables"]
        context = verify_call_context(variables["call_context"])
        self.assertEqual(variables["provider"], "sip")
        self.assertEqual(variables["caller_number"], "+421900000001")
        self.assertEqual(variables["provider_call_id"], "telnyx-provider-call-id")
        self.assertEqual(variables["conversation_id"], "conv_test")
        self.assertEqual(variables["client_name"], "Ján Novák")
        self.assertEqual(context["caller_phone"], "+421900000001")
        self.assertEqual(main.CALL_CONTEXT["sip-dialog-id"], "+421900000001")
        self.assertEqual(main.CONVERSATION_CONTEXT["conv_test"], "+421900000001")

    def test_untrusted_webhook_does_not_expose_or_store_identity(self):
        body = {
            "caller_id": "+421900000001",
            "call_id": "spoofed-call-id",
            "conversation_id": "conv_spoofed",
        }
        result = asyncio.run(main.prompt_config(self.request(body)))

        variables = result["dynamic_variables"]
        self.assertEqual(variables["caller_number"], "")
        self.assertEqual(variables["call_context"], "")
        self.assertEqual(variables["client_name"], "")
        self.assertEqual(main.CALL_CONTEXT, {})
        self.assertEqual(main.CONVERSATION_CONTEXT, {})


if __name__ == "__main__":
    unittest.main()
