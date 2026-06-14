pub trait Tokenizer: Send + Sync + Clone {
    /// Load the tokenizer from the provided path.
    fn new(tokenizer_path: &str) -> Result<Self, String>
    where
        Self: Sized;

    /// Encode a string into a list of token identifiers.
    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32>;

    /// Decode a list of token identifiers into a string.
    fn decode(&self, tokens: &[u32]) -> String;

    /// Raw bytes for `tokens`, not guaranteed to be valid UTF-8 on their own: byte-level BPE
    /// tokenizers routinely split a multi-byte character across tokens, so per-token byte chunks
    /// must be reassembled by the caller (the framework's `Utf8Buffer`). The default suits
    /// tokenizers whose per-token `decode` is already total; tokenizers whose `decode` can fail on
    /// a partial character (e.g. Tiktoken) must override this with a byte-level decode.
    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.decode(tokens).into_bytes()
    }

    /// Beginning of sentence token.
    fn bos(&self) -> String {
        self.decode(&[self.bos_id()])
    }

    /// Beginning of sentence token identifier.
    fn bos_id(&self) -> u32;

    /// End of sentence token.
    fn eos(&self) -> String {
        self.decode(&[self.eos_id()])
    }

    /// End of sentence token identifier.
    fn eos_id(&self) -> u32;

    /// Stop token identifiers.
    fn stop_ids(&self) -> Vec<u32>;

    /// Number of tokens needed as context for incremental streaming decoding.
    /// Default is 0 (no context/buffering needed).
    fn streaming_context_size(&self) -> usize {
        0
    }
}
