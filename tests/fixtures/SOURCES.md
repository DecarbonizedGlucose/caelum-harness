# Fixture sources

All captures are from the DeepSeek and OpenRouter endpoints configured in
`.env`; keys are not stored in fixtures.

| Protocol | Files | Source | Kind |
|---|---|---|---|
| OpenAI Chat | `openai_chat/deepseek_*` | `https://api.deepseek.com/chat/completions`, model `deepseek-flash` | real capture, 2026-11-21 |
| OpenAI Chat | `openai_chat/openrouter_*` | `https://openrouter.ai/api/v1/chat/completions`, model `mistralai/mistral-nemo` | real capture, 2026-11-21 |
| OpenAI Responses | `openai_responses/deepseek_*` | `https://api.deepseek.com/responses`, model `deepseek-flash` | real capture, 2026-11-21 |
| OpenAI Responses | `openai_responses/openrouter_text.json` | `https://openrouter.ai/api/v1/responses`, model `mistralai/mistral-nemo` | real capture, 2026-11-21 |
| Anthropic | `anthropic/deepseek_*` | `https://api.deepseek.com/anthropic/v1/messages`, model `deepseek-flash` | real capture, 2026-11-21 |
| Gemini | `gemini/*` | Google Gemini API documentation and `pi_agent_rust` VCR shapes | reference only, not a live capture |

## Notes

- DeepSeek's Anthropic-compatible signature is the message id; it is not an
  Anthropic-generated signature. It is still useful for parsing and replay
  shape, but not for signature validation.
- No official OpenAI, Anthropic, or Google API keys are available. Official
  endpoint recordings should be added when keys are available.
- `pi_agent_rust/tests/fixtures/vcr/` was used as a reference for field shapes.
  Its license is MIT with an OpenAI/Anthropic rider; no files were copied
  verbatim.
- Tests distinguish real captures from reference fixtures by fixture name and
  this table. Reference fixtures verify our interpretation, not provider
  behavior.
