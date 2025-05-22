use comfy_table::{Cell, CellAlignment, Table};
use std::{collections::BTreeSet, time::Duration};

pub const STATS_MARKER: &str = "##### BurnLM Stats";

/// A statistic entry returned by a Completion
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum StatEntry {
    /// Total inference duration
    InferenceDuration(Duration),
    /// Duration to download model
    ModelDownloadingDuration(Duration),
    /// Duration to load a model
    ModelLoadingDuration(Duration),
    /// A named stat
    Named(String, String),
    /// Total number of tokens
    TokensCount(usize),
    /// The number of tokens per second
    TokensPerSecond(usize, Duration),
    /// A total duration
    TotalDuration(Duration),
}

impl StatEntry {
    /// Return duration value if available
    pub fn get_duration(&self) -> Option<Duration> {
        match self {
            StatEntry::InferenceDuration(duration)
            | StatEntry::ModelDownloadingDuration(duration)
            | StatEntry::TotalDuration(duration)
            | StatEntry::TokensPerSecond(_, duration)
            | StatEntry::ModelLoadingDuration(duration) => Some(*duration),
            _ => None,
        }
    }
}

#[derive(Default)]
pub struct Stats {
    pub entries: BTreeSet<StatEntry>,
}

impl Stats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a markdown table displaying the stats.
    pub fn display_stats(&self) -> String {
        let mut table = Table::new();
        table
            .load_preset(comfy_table::presets::ASCII_MARKDOWN)
            .set_content_arrangement(comfy_table::ContentArrangement::Dynamic)
            .set_header(vec!["Statistic Name", "Value"]);
        for stat in &self.entries {
            let (stat_label, stat_value) = match stat {
                StatEntry::TokensPerSecond(token_count, duration) => {
                    let seconds = duration.as_secs_f64();
                    let value = if seconds > 0.0 {
                        // Compute tokens per second.
                        format!("{:.2}", (*token_count as f64) / seconds)
                    } else {
                        "N/A".to_string()
                    };
                    ("Tokens Per Second".to_string(), value)
                }
                StatEntry::TokensCount(count) => ("Tokens Count".to_string(), count.to_string()),
                StatEntry::InferenceDuration(duration) => (
                    "Inference Duration".to_string(),
                    format!("{:.2}s", duration.as_secs_f64()),
                ),
                StatEntry::ModelDownloadingDuration(duration) => (
                    "Model Downloading Duration".to_string(),
                    format!("{:.2}s", duration.as_secs_f64()),
                ),
                StatEntry::ModelLoadingDuration(duration) => (
                    "Model Loading Duration".to_string(),
                    format!("{:.2}s", duration.as_secs_f64()),
                ),
                StatEntry::TotalDuration(duration) => (
                    "Total Duration".to_string(),
                    format!("{:.2}s", duration.as_secs_f64()),
                ),
                StatEntry::Named(name, val) => (name.clone(), val.clone()),
            };

            table.add_row(vec![
                Cell::new(stat_label).set_alignment(CellAlignment::Left),
                Cell::new(stat_value).set_alignment(CellAlignment::Right),
            ]);
        }
        format!("\n{STATS_MARKER}\n{}", table)
    }
}
