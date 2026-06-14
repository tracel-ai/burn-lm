//! Incremental UTF-8 assembly for byte-level streaming detokenization.
//!
//! Byte-level BPE tokenizers (e.g. Llama-3's Tiktoken) routinely split a multi-byte character
//! across tokens, so decoding token-by-token yields byte chunks that are not individually valid
//! UTF-8. `Utf8Buffer` turns such a byte stream into a text stream. Each `push` emits the longest
//! valid prefix and holds back a trailing incomplete sequence — at most 3 bytes — until later bytes
//! complete it. `finish` drains whatever remains at the true end of the stream, where lossy
//! replacement with U+FFFD is permitted.
//!
//! Bytes that are definitely invalid, meaning no future byte can complete them, are replaced with
//! U+FFFD immediately in `push` rather than held back. Holding them back would stall all later text
//! behind bytes that can never become valid, and would break the 3-byte hold-back bound. For
//! in-vocab tokenizer output this never happens; only boundary splits do.

/// A streaming UTF-8 assembler. It holds at most 3 pending bytes between pushes, which is the
/// longest a partial UTF-8 sequence can run before its final byte arrives.
#[derive(Debug, Default)]
pub struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The number of held-back bytes, at most 3 for any input that is a prefix of valid UTF-8.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Append bytes and return all the text that is now complete, holding back a trailing incomplete
    /// sequence for a later push. This never panics, and never emits U+FFFD for bytes that a later
    /// push could still complete.
    pub fn push(&mut self, bytes: &[u8]) -> Option<String> {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    // Re-slice through the checked API rather than an unsafe `from_utf8_unchecked`.
                    out.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("prefix up to valid_up_to() is valid UTF-8"),
                    );
                    match err.error_len() {
                        // Definitely invalid: no future byte can fix it.
                        Some(len) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + len);
                        }
                        // Incomplete trailing sequence: hold it back.
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// End of stream: drain whatever bytes remain, replacing genuinely invalid or incomplete ones
    /// with U+FFFD. This is the one place that lossy replacement is allowed, because a held-back
    /// partial character can never be completed once the stream has ended.
    pub fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Push a multi-byte character split at every possible byte boundary; text must come out
    /// complete, no U+FFFD, nothing lost.
    fn assert_split_reassembles(ch: char) {
        let mut encoded = [0u8; 4];
        let bytes = ch.encode_utf8(&mut encoded).as_bytes();
        for split in 1..bytes.len() {
            let mut buf = Utf8Buffer::new();
            let first = buf.push(&bytes[..split]);
            assert_eq!(
                first, None,
                "incomplete prefix of {ch:?} split at {split} must be held back"
            );
            assert!(buf.pending_len() <= 3);
            let second = buf.push(&bytes[split..]);
            assert_eq!(second.as_deref(), Some(ch.to_string().as_str()));
            assert_eq!(buf.finish(), None, "nothing left to flush");
        }
    }

    #[test]
    fn two_byte_char_split_at_every_boundary() {
        assert_split_reassembles('é'); // 2 bytes
    }

    #[test]
    fn three_byte_char_split_at_every_boundary() {
        assert_split_reassembles('€'); // 3 bytes
    }

    #[test]
    fn four_byte_char_split_at_every_boundary() {
        assert_split_reassembles('🦀'); // 4 bytes
    }

    #[test]
    fn byte_at_a_time_stream_reassembles_mixed_text() {
        let text = "a é € 🦀 z";
        let mut buf = Utf8Buffer::new();
        let mut out = String::new();
        for byte in text.as_bytes() {
            if let Some(chunk) = buf.push(&[*byte]) {
                out.push_str(&chunk);
            }
            assert!(buf.pending_len() <= 3);
        }
        if let Some(rest) = buf.finish() {
            out.push_str(&rest);
        }
        assert_eq!(out, text);
    }

    #[test]
    fn valid_text_passes_through_whole() {
        let mut buf = Utf8Buffer::new();
        assert_eq!(buf.push("hello".as_bytes()).as_deref(), Some("hello"));
        assert_eq!(buf.pending_len(), 0);
    }

    #[test]
    fn empty_push_and_empty_finish_yield_none() {
        let mut buf = Utf8Buffer::new();
        assert_eq!(buf.push(&[]), None);
        assert_eq!(buf.finish(), None);
    }

    #[test]
    fn complete_text_before_incomplete_tail_is_emitted() {
        let mut buf = Utf8Buffer::new();
        // "ab" + first byte of '€'
        let mut bytes = b"ab".to_vec();
        bytes.push(0xE2);
        assert_eq!(buf.push(&bytes).as_deref(), Some("ab"));
        assert_eq!(buf.pending_len(), 1);
        // remaining two bytes of '€'
        assert_eq!(buf.push(&[0x82, 0xAC]).as_deref(), Some("€"));
    }

    #[test]
    fn finish_flushes_trailing_partial_char_lossily() {
        let mut buf = Utf8Buffer::new();
        assert_eq!(buf.push(&[0xF0, 0x9F]), None); // half a 🦀
        let flushed = buf.finish().expect("partial bytes must flush");
        assert!(flushed.contains('\u{FFFD}'));
        assert_eq!(buf.finish(), None, "finish drains");
    }

    #[test]
    fn definitely_invalid_byte_is_replaced_immediately_not_held() {
        let mut buf = Utf8Buffer::new();
        // 0xFF can never start a valid sequence; text after it must flow.
        let out = buf.push(&[b'a', 0xFF, b'b']).expect("text must flow");
        assert_eq!(out, "a\u{FFFD}b");
        assert_eq!(buf.pending_len(), 0);
    }

    #[test]
    fn invalid_continuation_does_not_stall_later_text() {
        let mut buf = Utf8Buffer::new();
        // 0xE2 expects two continuation bytes; 'x' proves it dead.
        let out = buf.push(&[0xE2, b'x']).expect("must emit");
        assert_eq!(out, "\u{FFFD}x");
        // Buffer stays usable afterwards.
        assert_eq!(buf.push("ok".as_bytes()).as_deref(), Some("ok"));
    }
}
