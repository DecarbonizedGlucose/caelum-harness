use serde_json::{Map, Value};

use crate::error::HarnessError;
use crate::model::message::ImageSource;
use crate::model::protocol::Protocol;

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

fn encode_base64(data: &[u8]) -> String {
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
