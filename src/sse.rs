use std::{
    pin::Pin,
    task::{Context, Poll},
};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use reqwest::Result;

/// A stream of SSE `data` fields obtained from a stream of bytes. Ignores the `event` field.
///
/// For SSE, the newline pair, not the TCP/HTTP chunk boundary, is the event boundary.
///
/// A naive SSE deserialization might try to convert each chunk of bytes into SSE events. However,
/// an SSE event can be spreaded over multiple chunks.
pub struct SseStream {
    /// A stream of bytes, could be obtained from [`reqwest::Response::bytes_stream`].
    stream: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
    buffer: String,
}

impl SseStream {
    pub fn new(stream: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>) -> Self {
        Self {
            stream,
            buffer: String::new(),
        }
    }

    /// Get the next event from the buffer if there is one in
    fn next_from_buffer(&mut self) -> Option<String> {
        if let Some(event) = self.first_event() {
            for line in event.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    return Some(data.to_owned());
                }
            }
        }
        // We might have split of an event, but did not find a data field. That is fine.
        None
    }

    /// The first event in the buffer, including the new lines
    fn first_event(&mut self) -> Option<String> {
        let position = self.buffer.find("\n\n")?;
        let event = self.buffer.drain(..position + 2).collect();
        Some(event)
    }
}

impl Stream for SseStream {
    type Item = Result<String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Do we still have events in the buffer? This can happen if one byte chunk contained
        // multiple sse events
        if let Some(event) = self.next_from_buffer() {
            return Poll::Ready(Some(Ok(event)));
        }

        // Nothing in the buffer, poll the underlying stream
        match self.stream.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let item = String::from_utf8_lossy(&chunk).replace("\r\n", "\n");
                self.buffer.push_str(&item);
                if let Some(event) = self.next_from_buffer() {
                    // We can report an entire event
                    Poll::Ready(Some(Ok(event)))
                } else {
                    // While we have received data from the stream, it has not been an entire event.
                    // Continue polling for more data
                    self.poll_next(cx)
                }
            }
            // Errors are forwarded
            Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
            // We are waiting for more data from the underlying stream
            Poll::Pending => Poll::Pending,
            // We have nothing in the buffer and the underlying stream is empty. We are done.
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    async fn one_sse_in_one_chunk() {
        // Given a byte stream with that contains an SSE event in the first stream item
        let chunk = "event: message\ndata: 42\n\n";
        let stream = futures_util::stream::once(async move { Ok(Bytes::from(chunk)) });
        let sse = SseStream::new(Box::pin(stream));

        // When collecting the events into a vec
        let mut events = sse.collect::<Vec<_>>().await;

        // Then we get the payload
        assert_eq!(events.len(), 1);
        assert_eq!(events.remove(0).unwrap(), "42");
    }

    #[tokio::test]
    async fn one_sse_over_multiple_chunks() {
        // Given a SSE event split across two chunks
        let chunks = vec!["event: message\ndata: 4", "2\n\n"];
        let stream =
            futures_util::stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))));
        let sse = SseStream::new(Box::pin(stream));

        // When collecting the events into a vec
        let mut events = sse.collect::<Vec<_>>().await;

        // Then we get the payload
        assert_eq!(events.len(), 1);
        assert_eq!(events.remove(0).unwrap(), "42");
    }

    #[tokio::test]
    async fn multiple_sse_in_one_chunk() {
        // Given a SSE event split across two chunks
        let chunk = "event: message\ndata: 42\n\nevent: message\ndata: 56\n\n";
        let stream = futures_util::stream::once(async move { Ok(Bytes::from(chunk)) });
        let sse = SseStream::new(Box::pin(stream));

        // When collecting the events into a vec
        let mut events = sse.collect::<Vec<_>>().await;

        // Then we get the payload
        assert_eq!(events.len(), 2);
        assert_eq!(events.remove(0).unwrap(), "42");
        assert_eq!(events.remove(0).unwrap(), "56");
    }

    #[tokio::test]
    async fn crlf_in_data() {
        // Given a SSE event with crtlf line endings
        let sse_event = "event: message\r\ndata: 123\r\n\r\n";
        let stream = futures_util::stream::once(async move { Ok(Bytes::from(sse_event)) });
        let sse = SseStream::new(Box::pin(stream));

        // When collecting the events
        let mut events = sse.collect::<Vec<_>>().await;

        // Then we get the payload
        assert_eq!(events.len(), 1);
        assert_eq!(events.remove(0).unwrap(), "123");
    }

    #[tokio::test]
    async fn incomplete_event_yields_no_events() {
        // Given a SSE event that is not complete
        let sse_event = "event: message\r\ndata: 1";
        let stream = futures_util::stream::once(async move { Ok(Bytes::from(sse_event)) });
        let mut sse = SseStream::new(Box::pin(stream));

        // When collecting the events
        let events = sse.next().await;

        // Then we get an empty vec
        assert!(events.is_none());
    }
}
