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
}

impl TraceContext {
    /// Construct a new trace context.
    ///
    /// Version 0 of the trace context specification only supports the `sampled` flag,
    /// so we can offer a bool input for this flag.
    ///
    /// [W3C TraceContext specification]: https://www.w3.org/TR/trace-context-2/#design-overview
    pub fn new(trace_id: u128, span_id: u64, sampled: bool) -> Self {
        Self {
            trace_id,
            span_id,
            sampled,
        }
    }

    /// Construct a new trace context with the `sampled` flag set to true.
    pub fn new_sampled(trace_id: u128, span_id: u64) -> Self {
        Self::new(trace_id, span_id, true)
    }

    /// Construct a new trace context with the `sampled` flag set to false.
    pub fn new_unsampled(trace_id: u128, span_id: u64) -> Self {
        Self::new(trace_id, span_id, false)
    }

    /// The version of the trace context specification that we support.
    ///
    /// https://www.w3.org/TR/trace-context-2/#version
    const SUPPORTED_VERSION: u8 = 0;

    pub fn traceparent(&self) -> String {
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
        let trace_context = TraceContext::new_sampled(trace_id, span_id);
        assert_eq!(trace_context.trace_flags(), 0x01);
    }

    #[test]
    fn trace_flags_if_not_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_unsampled(trace_id, span_id);
        assert_eq!(trace_context.trace_flags(), 0x00);
    }

    #[test]
    fn traceparent_generation_if_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_sampled(trace_id, span_id);
        assert_eq!(
            trace_context.traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        );
    }

    #[test]
    fn traceparent_generation_if_not_sampled() {
        let trace_id = 0x4bf92f3577b34da6a3ce929d0e0e4736;
        let span_id = 0x00f067aa0ba902b7;
        let trace_context = TraceContext::new_unsampled(trace_id, span_id);
        assert_eq!(
            trace_context.traceparent(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"
        );
    }
}
