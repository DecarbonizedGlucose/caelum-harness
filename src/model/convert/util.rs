use serde_json::{Map, Value, json};

use crate::error::HarnessError;
use crate::model::message::ImageSource;
use crate::model::protocol::Protocol;

pub(crate) fn is_done_payload(payload: &str) -> bool {
    payload.trim() == "[DONE]"
}

/// Parses tool arguments from a string or object into a JSON object.
pub(crate) fn decode_tool_arguments(
    value: &Value,
) -> Result<(Value, Option<String>, Option<String>), HarnessError> {
    match value {
        Value::String(raw) => {
            if raw.trim().is_empty() {
                return Ok((json!({}), Some("{}".to_string()), None));
            }
            Ok(match serde_json::from_str::<Value>(raw) {
                Ok(value) if value.is_object() => (value, Some(raw.clone()), None),
                Ok(_) => (
                    Value::Null,
                    Some(raw.clone()),
                    Some("tool arguments must be a JSON object".to_string()),
                ),
                Err(error) => (Value::Null, Some(raw.clone()), Some(error.to_string())),
            })
        }
        Value::Object(_) => {
            let raw = serde_json::to_string(value).map_err(|error| {
                HarnessError::internal(format!("could not serialize tool arguments: {error}"))
            })?;
            Ok((value.clone(), Some(raw), None))
        }
        Value::Null => Ok((json!({}), None, None)),
        _ => Ok((
            Value::Null,
            None,
            Some("tool arguments must be a JSON object or string".to_string()),
        )),
    }
}

/// Encodes tool arguments as the JSON string expected by OpenAI-style protocols.
pub(crate) fn encode_tool_arguments(
    args: &Value,
    raw_args: Option<&str>,
    args_error: Option<&str>,
) -> Result<String, HarnessError> {
    if args_error.is_some() {
        return Ok("{}".to_string());
    }
    if let Some(raw) = raw_args.filter(|raw| !raw.trim().is_empty()) {
        return Ok(raw.to_string());
    }
    if args.is_null() {
        return Ok("{}".to_string());
    }
    serde_json::to_string(args).map_err(|error| {
        HarnessError::internal(format!("could not serialize tool arguments: {error}"))
    })
}

pub(crate) fn as_object<'a>(
    value: &'a Value,
    protocol: Protocol,
    path: &str,
) -> Result<&'a Map<String, Value>, HarnessError> {
    value
        .as_object()
        .ok_or_else(|| HarnessError::protocol(protocol, path, "expected an object"))
}

pub(crate) fn required_str<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    protocol: Protocol,
    path: &str,
) -> Result<&'a str, HarnessError> {
    object.get(key).and_then(Value::as_str).ok_or_else(|| {
        HarnessError::protocol(protocol, format!("{path}.{key}"), "expected a string")
    })
}

pub(crate) fn data_url(source: &ImageSource) -> String {
    match source {
        ImageSource::Url(url) => url.clone(),
        ImageSource::Base64 { mime, data } => {
            format!("data:{mime};base64,{}", encode_base64(data))
        }
    }
}

pub(crate) fn encode_base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let value = (b0 << 16) | (b1 << 8) | b2;

        out.push(TABLE[(value >> 18) as usize & 63] as char);
        out.push(TABLE[(value >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(value >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[value as usize & 63] as char
        } else {
            '='
        });
    }
    out
}
