use std::iter;

/// Trace context that is propagated through HTTP headers to enable distributed tracing.
///
/// Currently still missing support for tracestate, otherwise compliant with
/// https://www.w3.org/TR/trace-context-2/#design-overview, which standardizes how
/// context information is sent and modified between services.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct TraceContext {
    /// https://www.w3.org/TR/trace-context/#trace-id
    trace_id: u128,
    /// https://www.w3.org/TR/trace-context/#parent-id
    span_id: u64,
    /// https://www.w3.org/TR/trace-context/#sampled-flag
    sampled: bool,
    /// https://www.w3.org/TR/trace-context/#tracestate-header
    state: Option<String>,
}

impl TraceContext {
    /// Construct a new trace context.
    ///
    /// Version 0 of the trace context specification only supports the `sampled` flag,
    /// so we can offer a bool input for this flag.
    ///
    /// [W3C TraceContext specification]: https://www.w3.org/TR/trace-context-2/#design-overview
    pub fn new(trace_id: u128, span_id: u64, sampled: bool, state: Option<String>) -> Self {
        Self {
            trace_id,
            span_id,
            sampled,
            state,
        }
    }

    /// Construct a new trace context with the `sampled` flag set to true.
    pub fn new_sampled(trace_id: u128, span_id: u64, state: Option<String>) -> Self {
        Self::new(trace_id, span_id, true, state)
    }

    /// Construct a new trace context with the `sampled` flag set to false.
    pub fn new_unsampled(trace_id: u128, span_id: u64, state: Option<String>) -> Self {
        Self::new(trace_id, span_id, false, state)
    }

    /// Render the context as w3c trace context headers.
    ///
    /// <https://www.w3.org/TR/trace-context-2>
    pub fn as_w3c_headers(&self) -> impl Iterator<Item = (&'static str, String)> {
        let mut first = Some(("traceparent", self.traceparent()));
        let mut second = if let Some(state) = &self.state {
            // Vendors MUST accept empty tracestate headers but SHOULD avoid sending them.
            if !state.is_empty() {
                Some(("tracestate", state.clone()))
            } else {
                None
            }
        } else {
            None
        };

        // By returning an iterator, we avoid allocating a Vec/HashMap.
        let mut counter = 0;
        iter::from_fn(move || {
            counter += 1;
            if counter == 1 {
                first.take()
            } else if counter == 2 {
                second.take()
            } else {
                None
            }
        })
    }

    /// The version of the trace context specification that we support.
    ///
    /// https://www.w3.org/TR/trace-context-2/#version
    const SUPPORTED_VERSION: u8 = 0;

    fn traceparent(&self) -> String {
        format!(
            "{:02x}-{:032x}-{:016x}-{:02x}",
            Self::SUPPORTED_VERSION,
            self.trace_id,
            self.span_id,
            self.trace_flags()
        )
    }

    /// The trace flags of this context.
    ///
    /// Version 0 of the trace context specification only supports the `sampled` flag.
    /// [W3C TraceContext specification]: https://www.w3.org/TR/trace-context/#sampled-flag
    fn trace_flags(&self) -> u8 {
        if self.sampled {
            0x01
        } else {
            0x00
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TraceContext;

    #[test]
    fn trace_flags_if_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_sampled(trace_id, span_id, None);
        assert_eq!(trace_context.trace_flags(), 0x01);
    }

    #[test]
    fn trace_flags_if_not_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_unsampled(trace_id, span_id, None);
        assert_eq!(trace_context.trace_flags(), 0x00);
    }

    #[test]
    fn traceparent_generation_if_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_sampled(trace_id, span_id, None);
        assert_eq!(
            trace_context.traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        );
    }

    #[test]
    fn traceparent_generation_if_not_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_unsampled(trace_id, span_id, None);
        assert_eq!(
            trace_context.traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        );
    }

    #[test]
    fn headers_include_traceparent() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_sampled(trace_id, span_id, None);
        let mut headers = trace_context.as_w3c_headers();

        // get first header
        let header = headers.next().unwrap();
        assert_eq!(header.0, "traceparent");
        assert_eq!(header.1, trace_context.traceparent());
        assert!(headers.next().is_none());
    }

    #[test]
    fn non_empty_tracestate_is_included() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context =
            TraceContext::new_sampled(trace_id, span_id, Some("foo=bar".to_string()));
        let mut headers = trace_context.as_w3c_headers();

        // get first header
        let header = headers.next().unwrap();
        assert_eq!(header.0, "traceparent");
        assert_eq!(header.1, trace_context.traceparent());

        // get second header
        let header = headers.next().unwrap();
        assert_eq!(header.0, "tracestate");
        assert_eq!(header.1, "foo=bar");
        assert!(headers.next().is_none());
    }
}
