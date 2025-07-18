use std::{borrow::Cow, pin::Pin, time::Duration};

use bytes::Bytes;
use futures_util::{stream::StreamExt, Stream};
use reqwest::{header, ClientBuilder, RequestBuilder, Response, StatusCode};
use serde::Deserialize;
use thiserror::Error as ThisError;
use tokenizers::Tokenizer;

use crate::{sse::SseStream, How, StreamJob, TraceContext};
use async_stream::stream;

/// A job send to the Aleph Alpha Api using the http client. A job wraps all the knowledge required
/// for the Aleph Alpha API to specify its result. Notably it includes the model(s) the job is
/// executed on. This allows this trait to hold in the presence of services, which use more than one
/// model and task type to achieve their result. On the other hand a bare [`crate::TaskCompletion`]
/// can not implement this trait directly, since its result would depend on what model is chosen to
/// execute it. You can remedy this by turning completion task into a job, calling
/// [`Task::with_model`].
pub trait Job {
    /// Output returned by [`crate::Client::output_of`]
    type Output;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de>;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output;
}

/// A task send to the Aleph Alpha Api using the http client. Requires to specify a model before it
/// can be executed.
pub trait Task {
    /// Output returned by [`crate::Client::output_of`]
    type Output;

    /// Expected answer of the Aleph Alpha API
    type ResponseBody: for<'de> Deserialize<'de>;

    /// Prepare the request for the Aleph Alpha API. Authentication headers can be assumed to be
    /// already set.
    fn build_request(&self, client: &reqwest::Client, base: &str, model: &str) -> RequestBuilder;

    /// Parses the response of the server into higher level structs for the user.
    fn body_to_output(&self, response: Self::ResponseBody) -> Self::Output;

    /// Turn your task into [`Job`] by annotating it with a model name.
    fn with_model<'a>(&'a self, model: &'a str) -> MethodJob<'a, Self>
    where
        Self: Sized,
    {
        MethodJob { model, task: self }
    }
}

/// Enriches the `Task` to a `Job` by appending the model it should be executed with. Use this as
/// input for [`Client::output_of`].
pub struct MethodJob<'a, T> {
    /// Name of the Aleph Alpha Model. E.g. "luminous-base".
    pub model: &'a str,
    /// Task to be executed against the model.
    pub task: &'a T,
}

impl<T> Job for MethodJob<'_, T>
where
    T: Task,
{
    type Output = T::Output;

    type ResponseBody = T::ResponseBody;

    fn build_request(&self, client: &reqwest::Client, base: &str) -> RequestBuilder {
        self.task.build_request(client, base, self.model)
    }

    fn body_to_output(&self, response: T::ResponseBody) -> T::Output {
        self.task.body_to_output(response)
    }
}

/// Sends HTTP request to the Aleph Alpha API
pub struct HttpClient {
    base: String,
    http: reqwest::Client,
    api_token: Option<String>,
}

impl HttpClient {
    /// In production you typically would want set this to <https://inference-api.pharia.your-company.com>.
    /// Yet you may want to use a different instance for testing.
    pub fn new(host: String, api_token: Option<String>) -> Result<Self, Error> {
        let http = ClientBuilder::new().build()?;

        Ok(Self {
            base: host,
            http,
            api_token,
        })
    }

    /// Construct and execute a request building on top of a `RequestBuilder`
    async fn response(&self, builder: RequestBuilder, how: &How) -> Result<Response, Error> {
        let query = if how.be_nice {
            [("nice", "true")].as_slice()
        } else {
            // nice=false is default, so we just omit it.
            [].as_slice()
        };

        let api_token = how
            .api_token
            .as_ref()
            .or(self.api_token.as_ref())
            .expect("API token needs to be set on client construction or per request");
        let mut builder = builder
            .query(query)
            .header(header::AUTHORIZATION, Self::header_from_token(api_token))
            .timeout(how.client_timeout);

        if let Some(trace_context) = &how.trace_context {
            for (key, value) in trace_context.as_w3c_headers() {
                builder = builder.header(key, value);
            }
        }

        let response = builder.send().await.map_err(|reqwest_error| {
            if reqwest_error.is_timeout() {
                Error::ClientTimeout(how.client_timeout)
            } else {
                reqwest_error.into()
            }
        })?;
        translate_http_error(response).await
    }

    /// Execute a task with the aleph alpha API and fetch its result.
    ///
    /// ```no_run
    /// use aleph_alpha_client::{Client, How, TaskCompletion, Task, Error};
    ///
    /// async fn print_completion() -> Result<(), Error> {
    ///     // Authenticate against API. Fetches token.
    ///     let client = Client::from_env()?;
    ///
    ///     // Name of the model we we want to use. Large models give usually better answer, but are
    ///     // also slower and more costly.
    ///     let model = "luminous-base";
    ///
    ///     // The task we want to perform. Here we want to continue the sentence: "An apple a day
    ///     // ..."
    ///     let task = TaskCompletion::from_text("An apple a day");
    ///
    ///     // Retrieve answer from API
    ///     let response = client.output_of(&task.with_model(model), &How::default()).await?;
    ///
    ///     // Print entire sentence with completion
    ///     println!("An apple a day{}", response.completion);
    ///     Ok(())
    /// }
    /// ```
    pub async fn output_of<T: Job>(&self, task: &T, how: &How) -> Result<T::Output, Error> {
        let builder = task.build_request(&self.http, &self.base);
        let response = self.response(builder, how).await?;
        let response_body: T::ResponseBody = response.json().await?;
        let answer = task.body_to_output(response_body);
        Ok(answer)
    }

    /// Execute a stream task with the aleph alpha API and stream its result.
    pub async fn stream_output_of<'task, T: StreamJob + Send + Sync + 'task>(
        &self,
        task: T,
        how: &How,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<T::Output, Error>> + Send + 'task>>, Error>
    where
        T::Output: 'static,
    {
        let builder = task.build_request(&self.http, &self.base);
        let response = self.response(builder, how).await?;
        let stream = Box::pin(response.bytes_stream());
        Self::parse_stream_output(stream, task).await
    }

    /// Parse a stream of bytes into a stream of [`crate::StreamTask::Output`] objects.
    ///
    /// The [`crate::StreamTask::body_to_output`] allows each implementation to decide how to handle
    /// the response events.
    pub async fn parse_stream_output<'task, T: StreamJob + Send + Sync + 'task>(
        stream: Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>,
        task: T,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<T::Output, Error>> + Send + 'task>>, Error>
    where
        T::Output: 'static,
    {
        let mut stream = SseStream::new(stream);

        Ok(Box::pin(stream! {
            while let Some(item) = stream.next().await {
                match item {
                    Ok(data) => {
                        // The last stream event for the chat endpoint always is "[DONE]". We assume
                        // that the consumer of this library is not interested in this event.
                        if data.trim() == "[DONE]" {
                            break;
                        }
                        // Each task defines its response body as an associated type. This allows
                        // us to define generic parsing logic for multiple streaming tasks. In
                        // addition, tasks define an output type, which is a higher level
                        // abstraction over the response body. With the `body_to_output` method,
                        // tasks define logic to parse a response body into an output. This
                        // decouples the parsing logic from the data handed to users.
                        match serde_json::from_str::<T::ResponseBody>(&data) {
                            Ok(b) => yield Ok(task.body_to_output(b)),
                            Err(e) => {
                                yield Err(Error::InvalidStream {
                                    deserialization_error: e.to_string(),
                                });
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(e.into());
                    }
                }
            }
        }))
    }

    fn header_from_token(api_token: &str) -> header::HeaderValue {
        let mut auth_value = header::HeaderValue::from_str(&format!("Bearer {api_token}")).unwrap();
        // Consider marking security-sensitive headers with `set_sensitive`.
        auth_value.set_sensitive(true);
        auth_value
    }

    pub async fn tokenizer_by_model(
        &self,
        model: &str,
        api_token: Option<String>,
        context: Option<TraceContext>,
    ) -> Result<Tokenizer, Error> {
        let api_token = api_token
            .as_ref()
            .or(self.api_token.as_ref())
            .expect("API token needs to be set on client construction or per request");
        let mut builder = self
            .http
            .get(format!("{}/models/{model}/tokenizer", self.base))
            .header(header::AUTHORIZATION, Self::header_from_token(api_token));

        if let Some(trace_context) = &context {
            for (key, value) in trace_context.as_w3c_headers() {
                builder = builder.header(key, value);
            }
        }

        let response = builder.send().await?;
        let response = translate_http_error(response).await?;
        let bytes = response.bytes().await?;
        let tokenizer = Tokenizer::from_bytes(bytes).map_err(|e| Error::InvalidTokenizer {
            deserialization_error: e.to_string(),
        })?;
        Ok(tokenizer)
    }
}

async fn translate_http_error(response: reqwest::Response) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if !status.is_success() {
        // Store body in a variable, so we can use it, even if it is not an Error emitted by
        // the API, but an intermediate Proxy like NGinx, so we can still forward the error
        // message.
        let body = response.text().await?;
        // If the response is an error emitted by the API, this deserialization should succeed.
        let api_error: Result<ApiError, _> = serde_json::from_str(&body);
        let translated_error = match status {
            StatusCode::TOO_MANY_REQUESTS => Error::TooManyRequests,
            StatusCode::NOT_FOUND => {
                if api_error.is_ok_and(|error| error.code == "UNKNOWN_MODEL") {
                    Error::ModelNotFound
                } else {
                    Error::Http {
                        status: status.as_u16(),
                        body,
                    }
                }
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                // Presence of `api_error` implies the error originated from the API itself (rather
                // than the intermediate proxy) and so we can decode it as such.
                if api_error.is_ok_and(|error| error.code == "QUEUE_FULL") {
                    Error::Busy
                } else {
                    Error::Unavailable
                }
            }
            _ => Error::Http {
                status: status.as_u16(),
                body,
            },
        };
        Err(translated_error)
    } else {
        Ok(response)
    }
}

/// We are only interested in the status codes of the API.
#[derive(Deserialize, Debug)]
struct ApiError<'a> {
    /// Unique string in capital letters emitted by the API to signal different kinds of errors in a
    /// finer granularity then the HTTP status codes alone would allow for.
    ///
    /// E.g. Differentiating between request rate limiting and parallel tasks limiting which both
    /// are 429 (the former is emitted by NGinx though).
    code: Cow<'a, str>,
}

/// Errors returned by the Aleph Alpha Client
#[derive(ThisError, Debug)]
pub enum Error {
    #[error(
        "The model was not found. Please check the provided model name. You can query the list \
        of available models at the `models` endpoint. If you believe the model should be
        available, contact the operator of your inference server."
    )]
    ModelNotFound,
    /// User exceeds his current Task Quota.
    #[error(
        "You are trying to send too many requests to the API in to short an interval. Slow down a \
        bit, otherwise these error will persist. Sorry for this, but we try to prevent DOS attacks."
    )]
    TooManyRequests,
    /// Model is busy. Most likely due to many other users requesting its services right now.
    #[error(
        "Sorry the request to the Aleph Alpha API has been rejected due to the requested model \
        being very busy at the moment. We found it unlikely that your request would finish in a \
        reasonable timeframe, so it was rejected right away, rather than make you wait. You are \
        welcome to retry your request any time."
    )]
    Busy,
    /// The API itself is unavailable, most likely due to restart.
    #[error(
        "The service is currently unavailable. This is likely due to restart. Please try again \
        later."
    )]
    Unavailable,
    #[error("No response received within given timeout: {0:?}")]
    ClientTimeout(Duration),
    /// An error on the Http Protocol level.
    #[error("HTTP request failed with status code {}. Body:\n{}", status, body)]
    Http { status: u16, body: String },
    #[error(
        "Tokenizer could not be correctly deserialized. Caused by:\n{}",
        deserialization_error
    )]
    InvalidTokenizer { deserialization_error: String },
    /// Deserialization error of the stream event.
    #[error(
        "Stream event could not be correctly deserialized. Caused by:\n{}.",
        deserialization_error
    )]
    InvalidStream { deserialization_error: String },
    /// Most likely either TLS errors creating the Client, or IO errors.
    #[error(transparent)]
    Other(#[from] reqwest::Error),
}

#[cfg(test)]
mod tests {
    use crate::{ChatEvent, CompletionEvent, Message, TaskChat, TaskCompletion};

    use super::*;

    #[tokio::test]
    async fn stream_chunk_event_is_parsed() {
        // Given a completion task and part of its response stream that includes a stream chunk
        let task = TaskCompletion::from_text("An apple a day");
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"type\":\"stream_chunk\",\"index\":0,\"completion\":\" The New York Times, May 15\"}\n\ndata: [DONE]";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a completion event is yielded
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events.remove(0).unwrap(), CompletionEvent::Delta { completion, .. } if completion == " The New York Times, May 15")
        );
    }

    #[tokio::test]
    async fn completion_summary_event_is_parsed() {
        // Given a completion task and part of its response stream that includes a finish reason and a summary
        let task = TaskCompletion::from_text("An apple a day");
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"type\":\"stream_summary\",\"index\":0,\"model_version\":\"2022-04\",\"finish_reason\":\"maximum_tokens\"}\n\ndata: {\"type\":\"completion_summary\",\"num_tokens_prompt_total\":1,\"num_tokens_generated\":7}\n\n";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a finish reason event and a summary event are yielded
        assert_eq!(events.len(), 2);
        assert!(
            matches!(events.remove(0).unwrap(), CompletionEvent::Finished { reason } if reason == "maximum_tokens")
        );
        assert!(
            matches!(events.remove(0).unwrap(), CompletionEvent::Summary { usage, .. } if usage.prompt_tokens == 1 && usage.completion_tokens == 7)
        );
    }

    #[tokio::test]
    async fn chat_usage_event_is_parsed() {
        // Given a chat task and part of its response stream that includes a usage event
        let task = TaskChat::with_messages(vec![Message::user("An apple a day")]);
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"id\": \"67c5b5f2-6672-4b0b-82b1-cc844127b214\",\"choices\": [],\"created\": 1739539146,\"model\": \"pharia-1-llm-7b-control\",\"system_fingerprint\": \".unknown.\",\"object\": \"chat.completion.chunk\",\"usage\": {\"prompt_tokens\": 20,\"completion_tokens\": 10,\"total_tokens\": 30}}\n\n";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a summary event is yielded
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events.remove(0).unwrap(), ChatEvent::Summary { usage } if usage.prompt_tokens == 20 && usage.completion_tokens == 10)
        );
    }

    #[tokio::test]
    async fn chat_stream_chunk_with_role_is_parsed() {
        // Given a chat task and part of its response stream that includes a stream chunk with a role
        let task = TaskChat::with_messages(vec![Message::user("An apple a day")]);
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"id\":\"831e41b4-2382-4b08-990e-0a3859967f43\",\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"logprobs\":null}],\"created\":1729782822,\"model\":\"pharia-1-llm-7b-control\",\"system_fingerprint\":null,\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a message start event with a role is yielded
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events.remove(0).unwrap(), ChatEvent::MessageStart { role } if role == "assistant")
        );
    }

    #[tokio::test]
    async fn chat_stream_chunk_without_role_is_parsed() {
        // Given a chat task and part of its response stream that includes a pure content stream chunk
        let task = TaskChat::with_messages(vec![Message::user("An apple a day")]);
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"id\":\"a3ceca7f-32b2-4a6c-89e7-bc8eb5327f76\",\"choices\":[{\"finish_reason\":null,\"index\":0,\"delta\":{\"content\":\"Hello! How can I help you today? If you have any questions or need assistance, feel free to ask.\"},\"logprobs\":null}],\"created\":1729784197,\"model\":\"pharia-1-llm-7b-control\",\"system_fingerprint\":null,\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a message delta event with content is yielded
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events.remove(0).unwrap(), ChatEvent::MessageDelta { content, logprobs } if content == "Hello! How can I help you today? If you have any questions or need assistance, feel free to ask." && logprobs.is_empty())
        );
    }

    #[tokio::test]
    async fn chat_stream_chunk_without_content_but_with_finish_reason_is_parsed() {
        // Given a chat task and part of its response stream that includes a stream chunk with a finish reason
        let task = TaskChat::with_messages(vec![Message::user("An apple a day")]);
        let job = task.with_model("pharia-1-llm-7b-control");
        let bytes = "data: {\"id\":\"a3ceca7f-32b2-4a6c-89e7-bc8eb5327f76\",\"choices\":[{\"finish_reason\":\"stop\",\"index\":0,\"delta\":{},\"logprobs\":null}],\"created\":1729784197,\"model\":\"pharia-1-llm-7b-control\",\"system_fingerprint\":null,\"object\":\"chat.completion.chunk\",\"usage\":null}\n\n";
        let stream = Box::pin(futures_util::stream::once(
            async move { Ok(Bytes::from(bytes)) },
        ));

        // When converting it to a stream of events
        let stream = HttpClient::parse_stream_output(stream, job).await.unwrap();
        let mut events = stream.collect::<Vec<_>>().await;

        // Then a message end event with a stop reason is yielded
        assert_eq!(events.len(), 1);
        assert!(
            matches!(events.remove(0).unwrap(), ChatEvent::MessageEnd { stop_reason } if stop_reason == "stop")
        );
    }
}
