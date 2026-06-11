use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;

use super::Tokenizer;

const BOS_TOKEN: &str = "<|begin_of_text|>";
const EOS_TOKEN: &str = "<|end_of_text|>";
const EOT_TOKEN: &str = "<|eot_id|>";
const EOM_TOKEN: &str = "<|eom_id|>";

const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;
const SPECIAL_TOKENS: [&str; 11] = [
    BOS_TOKEN,
    EOS_TOKEN,
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    EOM_TOKEN, // end of message
    EOT_TOKEN, // end of turn
    "<|python_tag|>",
];
const PATTERN: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;

#[derive(Debug, Clone)]
pub struct Tiktoken {
    bpe: CoreBPE,
    bos_token_id: usize,
    eos_token_id: usize,
    eot_token_id: usize,
    eom_token_id: usize,
}

impl Tokenizer for Tiktoken {
    /// Load the [Tiktoken](https://github.com/openai/tiktoken) tokenizer.
    fn new(tiktoken_bpe_file: &str) -> Result<Self, String> {
        let file = File::open(tiktoken_bpe_file).map_err(|e| e.to_string())?;
        let mut mergeable_ranks: HashMap<Vec<u8>, usize> = HashMap::default();

        for line in BufReader::new(file).lines() {
            let line = match line {
                Ok(val) => val,
                Err(err) => return Err(err.to_string()),
            };
            let mut parts = line.split(' ');
            let token = STANDARD
                .decode(parts.next().ok_or("Missing token")?)
                .map_err(|e| e.to_string())?;
            let rank = parts
                .next()
                .ok_or("Missing rank")?
                .parse::<usize>()
                .map_err(|e| e.to_string())?;

            mergeable_ranks.insert(token, rank);
        }
        let num_base_tokens = mergeable_ranks.len();

        let special_tokens = [
            SPECIAL_TOKENS
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
            (0..NUM_RESERVED_SPECIAL_TOKENS - SPECIAL_TOKENS.len())
                .map(|i| format!("<|reserved_special_token_{}|>", i + 2))
                .collect::<Vec<_>>(),
        ]
        .concat();
        let special_tokens = special_tokens
            .into_iter()
            .enumerate()
            .map(|(i, s)| (s, i + num_base_tokens))
            .collect::<HashMap<String, usize>>();

        let bos_token_id = special_tokens[BOS_TOKEN];
        let eos_token_id = special_tokens[EOS_TOKEN];
        let eot_token_id = special_tokens[EOT_TOKEN];
        let eom_token_id = special_tokens[EOM_TOKEN];

        let bpe =
            CoreBPE::new(mergeable_ranks, special_tokens, PATTERN).map_err(|e| e.to_string())?;
        Ok(Self {
            bpe,
            bos_token_id,
            eos_token_id,
            eot_token_id,
            eom_token_id,
        })
    }

    fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<u32> {
        let bos_token = if bos { vec![self.bos_token_id] } else { vec![] };
        let eos_token = if eos { vec![self.eos_token_id] } else { vec![] };

        let tokens = self.bpe.encode_with_special_tokens(text);

        [bos_token, tokens, eos_token]
            .into_iter()
            .flat_map(|t| t.into_iter())
            .map(|t| t as u32)
            .collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.bpe
            .decode(tokens.iter().map(|&t| t as usize).collect())
            .expect("Should decode tokens")
    }

    /// Byte-level decode: `CoreBPE::_decode_native` (public despite the name) never fails for
    /// in-vocab ids, so the per-token "split UTF-8 character" panic class of
    /// [`decode`](Self::decode) cannot occur here. The returned bytes are reassembled into text
    /// by the framework's `Utf8Buffer`.
    fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        self.bpe
            ._decode_native(&tokens.iter().map(|&t| t as usize).collect::<Vec<_>>())
    }

    fn bos_id(&self) -> u32 {
        self.bos_token_id as u32
    }

    fn eos_id(&self) -> u32 {
        self.eos_token_id as u32
    }

    fn stop_ids(&self) -> Vec<u32> {
        vec![
            self.eos_id(),
            self.eom_token_id as u32,
            self.eot_token_id as u32,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal byte-level BPE vocab (all 256 single bytes, no merges) so the tokenizer
    /// splits any multi-byte character across tokens.
    fn byte_level_tiktoken() -> Tiktoken {
        let path = std::env::temp_dir().join(format!(
            "burn-lm-test-bpe-{}-{}.tiktoken",
            std::process::id(),
            std::thread::current().name().unwrap_or("t").len(),
        ));
        let mut file = File::create(&path).unwrap();
        for byte in 0..=255u8 {
            writeln!(file, "{} {}", STANDARD.encode([byte]), byte).unwrap();
        }
        drop(file);
        let tok = Tiktoken::new(path.to_str().unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        tok
    }

    /// The serving-panic killer: a multi-byte character split across tokens panics `decode` per
    /// token, but `decode_bytes` is total and the byte chunks reassemble into the exact original
    /// text.
    #[test]
    fn decode_bytes_is_total_on_split_multibyte_characters() {
        let tok = byte_level_tiktoken();
        let text = "a€🦀";
        let tokens = tok.encode(text, false, false);
        assert!(
            tokens.len() >= text.len(),
            "byte-level vocab must split multi-byte chars: {tokens:?}"
        );

        let mut bytes = Vec::new();
        for t in &tokens {
            // Per-token byte decode never fails, even mid-character.
            bytes.extend(tok.decode_bytes(&[*t]));
        }
        assert_eq!(bytes, text.as_bytes());
    }

    /// Documents why `decode_bytes` exists: per-token `decode` panics on a continuation byte
    /// (the in-generation panic that bricked the batching worker).
    #[test]
    fn per_token_decode_panics_on_split_utf8() {
        let tok = byte_level_tiktoken();
        let tokens = tok.encode("€", false, false);
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tok.decode(&tokens[..1])));
        assert!(result.is_err(), "decode of a partial char should panic");
    }
}
