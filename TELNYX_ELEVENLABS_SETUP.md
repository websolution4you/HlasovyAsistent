# Telnyx + ElevenLabs booking tools

## Server environment

Set these values to independent, randomly generated secrets:

- `VOICE_CALL_CONTEXT_SECRET`: signs short-lived caller context tokens.
- `ELEVENLABS_INIT_WEBHOOK_SECRET`: authenticates the ElevenLabs initiation webhook.

Keep the existing Supabase and tenant variables configured. Do not reuse a public API key for either secret.

## Conversation initiation webhook

Configure the NTC ElevenLabs agent to call:

`POST https://<backend-host>/api/prompt-config?tenant=ntc`

Add this static request header:

`X-ElevenLabs-Webhook-Secret: <ELEVENLABS_INIT_WEBHOOK_SECRET>`

The response supplies these dynamic variables:

- `caller_number`
- `provider_call_id`
- `call_sid` (compatibility alias)
- `conversation_id`
- `provider`
- `call_context`
- `client_name`
- `client_salutation`

The backend accepts Telnyx/SIP caller and call IDs from the initiation payload. It only exposes caller identity and signs `call_context` when the webhook secret is valid.

## ElevenLabs server tools

All tools use `POST` with a JSON body.

### Create booking

URL: `https://<backend-host>/api/ntc-create-booking`

Keep the existing booking parameters and add:

- `call_context`: `{{call_context}}`
- `caller_number`: `{{caller_number}}`
- `dynamic_variables.call_context`: `{{call_context}}` (optional compatibility copy)
- `dynamic_variables.call_sid`: `{{call_sid}}` (Twilio compatibility)

Registered-member wallet bookings require verified call context. Retries for the same call, member, sport and time use the same idempotency key.

### List upcoming bookings

URL: `https://<backend-host>/api/ntc-upcoming-bookings`

Body:

- `call_context`: `{{call_context}}`
- `call_sid`: `{{call_sid}}` (optional Twilio fallback)

### Cancel booking

URL: `https://<backend-host>/api/ntc-cancel-booking`

Body:

- `call_context`: `{{call_context}}`
- `call_sid`: `{{call_sid}}` (optional Twilio fallback)
- `booking_reference`: tool-selected booking reference
- `action`: `prepare` or `confirm`
- `confirmation_token`: token returned by `prepare`; required for `confirm`

The agent must ask the caller for explicit confirmation after `prepare` and only then call `confirm` with the returned token.

## Compatibility and security

- Existing Twilio calls continue to work through a validated `CA...` call SID stored by the inbound Twilio webhook.
- Telnyx/SIP sensitive tools require the signed `call_context`; an arbitrary provider call ID is not trusted.
- Context tokens expire after four hours and are valid across server processes.
- Cancellation remains scoped to the verified tenant, user, booking ID, future start time and `confirmed` status.
