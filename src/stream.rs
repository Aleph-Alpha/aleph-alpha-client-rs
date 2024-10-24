use crate::{completion::BodyCompletion, http::Task, Error, TaskCompletion};
use serde::Deserialize;

/// Describes a chunk of a completion stream
#[derive(Deserialize, Debug)]
pub struct StreamChunk {
    /// The index of the stream that this chunk belongs to.
    /// This is relevant if multiple completion streams are requested (see parameter n).
    pub index: u32,
    /// The completion of the stream.
    pub completion: String,
}

/// Denotes the end of a completion stream.
///
/// The index of the stream that is being terminated is not deserialized.
/// It is only relevant if multiple completion streams are requested, (see parameter n),
/// which is not supported by this crate yet.
#[derive(Deserialize, Debug)]
pub struct StreamSummary {
    /// Model name and version (if any) of the used model for inference.
    pub model_version: String,
    /// The reason why the model stopped generating new tokens.
    pub finish_reason: String,
}

/// Denotes the end of all completion streams.
#[derive(Deserialize, Debug)]
pub struct CompletionSummary {
    /// Number of tokens combined across all completion tasks.
    /// In particular, if you set best_of or n to a number larger than 1 then we report the
    /// combined prompt token count for all best_of or n tasks.
    pub num_tokens_prompt_total: u32,
    /// Number of tokens combined across all completion tasks.
    /// If multiple completions are returned or best_of is set to a value greater than 1 then
    /// this value contains the combined generated token count.
    pub num_tokens_generated: u32,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum Event {
    StreamChunk(StreamChunk),
    StreamSummary(StreamSummary),
    CompletionSummary(CompletionSummary),
}

pub struct TaskStreamCompletion<'a> {
    pub task: TaskCompletion<'a>,
}

impl Task for TaskStreamCompletion<'_> {
    type Output = Event;

    type ResponseBody = Event;

    fn build_request(
        &self,
        client: &reqwest::Client,
        base: &str,
        model: &str,
    ) -> reqwest::RequestBuilder {
        let body = BodyCompletion::new(model, &self.task).with_streaming();
        client.post(format!("{base}/complete")).json(&body)
    }

    fn body_to_output(response: Self::ResponseBody) -> Self::Output {
        response
    }
}

pub fn parse_stream_event<ResponseBody>(bytes: &[u8]) -> Vec<Result<ResponseBody, Error>>
where
    ResponseBody: for<'de> Deserialize<'de>,
{
    String::from_utf8_lossy(bytes)
        .split("data: ")
        .skip(1)
        .map(|s| {
            serde_json::from_str(s).map_err(|e| Error::StreamDeserializationError {
                cause: e.to_string(),
                event: s.to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_chunk_event_is_parsed() {
        // Given some bytes
        let bytes = b"data: {\"type\":\"stream_chunk\",\"index\":0,\"completion\":\" The New York Times, May 15\"}\n\n";

        // When they are parsed
        let events = parse_stream_event::<Event>(bytes);
        let event = events.first().unwrap().as_ref().unwrap();

        // Then the event is a stream chunk
        match event {
            Event::StreamChunk(chunk) => assert_eq!(chunk.index, 0),
            _ => panic!("Expected a stream chunk"),
        }
    }

    #[test]
    fn completion_summary_event_is_parsed() {
        // Given some bytes with a stream summary and a completion summary
        let bytes = b"data: {\"type\":\"stream_summary\",\"index\":0,\"model_version\":\"2022-04\",\"finish_reason\":\"maximum_tokens\"}\n\ndata: {\"type\":\"completion_summary\",\"num_tokens_prompt_total\":1,\"num_tokens_generated\":7}\n\n";

        // When they are parsed
        let events = parse_stream_event::<Event>(bytes);

        // Then the first event is a stream summary and the last event is a completion summary
        let first = events.first().unwrap().as_ref().unwrap();
        match first {
            Event::StreamSummary(summary) => assert_eq!(summary.finish_reason, "maximum_tokens"),
            _ => panic!("Expected a completion summary"),
        }
        let second = events.last().unwrap().as_ref().unwrap();
        match second {
            Event::CompletionSummary(summary) => assert_eq!(summary.num_tokens_generated, 7),
            _ => panic!("Expected a completion summary"),
        }
    }
}
