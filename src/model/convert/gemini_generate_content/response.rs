use crate::error::HarnessError;
use crate::model::convert::error::{decode_error, provider_error};
use crate::model::convert::gemini_generate_content::{
    THOUGHT_SIGNATURE_KIND, UsageAccumulator, is_protocol_failure_reason, map_finish_reason,
};
use crate::model::convert::util::decode_tool_arguments;
use crate::model::message::{
    Block, Message, Opaque, Origin, ReasoningBlock, TextBlock, ToolCallBlock,
};
use crate::model::protocol::Protocol;
use crate::model::stream::{Assembler, StreamEvent};
use crate::model::wire::gemini_generate_content as wire;

/// Decodes a Gemini GenerateContent response body.
pub(super) fn decode_response(raw: &str, origin: &Origin) -> Result<Message, HarnessError> {
    let response: wire::GenerateContentResponse = serde_json::from_str(raw)
        .map_err(|error| decode_error("Gemini response", Protocol::GeminiGenerateContent, error))?;

    if let Some(error) = &response.error {
        return Err(provider_error(
            Protocol::GeminiGenerateContent,
            error.message.as_deref(),
            error.status.as_deref(),
            None,
            error.code,
        ));
    }

    if response
        .prompt_feedback
        .as_ref()
        .and_then(|feedback| feedback.block_reason.as_deref())
        .is_some()
    {
        return Err(HarnessError::content_filter(
            Protocol::GeminiGenerateContent,
        ));
    }

    let candidate = response.candidates.first();
    let mut blocks = Vec::new();
    if let Some(content) = candidate.and_then(|candidate| candidate.content.as_ref()) {
        for (index, part) in content.parts.iter().enumerate() {
            if let Some(block) = decode_part(part, origin, index)? {
                blocks.push(block);
            }
        }
    }

    let raw_reason = candidate.and_then(|candidate| candidate.finish_reason.as_deref());
    if blocks.is_empty()
        && let Some(reason) = raw_reason
        && is_protocol_failure_reason(reason)
    {
        return Err(HarnessError::protocol(
            Protocol::GeminiGenerateContent,
            "candidates[0].finishReason",
            format!("generation failed with {reason}"),
        ));
    }

    let has_tool_calls = blocks
        .iter()
        .any(|block| matches!(block, Block::ToolCall(_)));
    let stop_reason = raw_reason.map(|reason| map_finish_reason(reason, has_tool_calls));
    let mut usage_acc = UsageAccumulator::default();
    let usage = usage_acc.update(response.usage_metadata.as_ref());

    let mut events = Vec::new();
    events.push(StreamEvent::MessageStart {
        response_id: response.response_id.clone(),
    });
    for (index, block) in blocks.into_iter().enumerate() {
        events.push(StreamEvent::CompleteBlock { index, block });
    }
    if let Some(usage) = usage {
        events.push(StreamEvent::Usage(usage));
    }
    events.push(StreamEvent::Finish { stop_reason });

    let mut assembler = Assembler::new(origin.clone());
    for event in events {
        assembler.push(event)?;
    }
    assembler.finish()
}

pub(super) fn signature_opaque(origin: &Origin, signature: Option<&str>) -> Option<Opaque> {
    signature.filter(|sig| !sig.is_empty()).map(|sig| Opaque {
        origin: origin.clone(),
        kind: THOUGHT_SIGNATURE_KIND.to_string(),
        data: sig.to_string(),
        item_id: None,
    })
}

fn decode_part(
    part: &wire::Part,
    origin: &Origin,
    index: usize,
) -> Result<Option<Block>, HarnessError> {
    let opaque = signature_opaque(origin, part.thought_signature.as_deref());

    if let Some(call) = &part.function_call {
        let (args, raw_args, args_error) = decode_tool_arguments(&call.args)?;
        return Ok(Some(Block::ToolCall(ToolCallBlock {
            id: call
                .id
                .as_deref()
                .filter(|id| !id.is_empty())
                .map(ToString::to_string)
                .unwrap_or_else(|| format!("call-{index}")),
            name: call.name.clone(),
            args,
            raw_args,
            args_error,
            provider_item_id: None,
            opaque,
        })));
    }

    if let Some(text) = part.text.as_deref() {
        let is_thought = part.thought.unwrap_or(false);
        if text.is_empty() && opaque.is_none() {
            return Ok(None);
        }
        if is_thought {
            return Ok(Some(Block::Reasoning(ReasoningBlock {
                text: text.to_string(),
                opaque,
            })));
        }
        return Ok(Some(Block::Text(TextBlock {
            text: text.to_string(),
            opaque,
        })));
    }

    Ok(None)
}
