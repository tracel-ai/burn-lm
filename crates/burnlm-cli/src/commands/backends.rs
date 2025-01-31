use std::collections::BTreeMap;

use comfy_table::{Cell, Table};
use strum::IntoEnumIterator;

use crate::backends::BackendValues;

pub(crate) fn create() -> clap::Command {
    clap::Command::new("backends").about("List all available Burn backends.")
}

pub(crate) fn handle() -> anyhow::Result<()> {
    let mut backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for backend in BackendValues::iter() {
        let backend_string = backend.to_string();
        let key = if backend_string.starts_with("candle") {
            "[5]Candle"
        } else if backend_string.starts_with("cuda") {
            "[2]Cuda"
        } else if backend_string.starts_with("hip") {
            "[3]ROCm HIP"
        } else if backend_string.starts_with("ndarray") {
            "[4]ndarray"
        } else if backend_string.starts_with("tch") {
            "[6]LibTorch"
        } else if backend_string.starts_with("wgpu") {
            "[1]WebGPU"
        } else {
            panic!("add support for backend: {backend_string}");
        };
        backends.entry(key.to_string()).or_default().push(backend_string);
    }

    // display the supported backends in a nice little table
    let mut table = Table::new();
    let header_cells: Vec<_> = backends.keys().map(|k| Cell::new(k.get(3..).unwrap_or("")).add_attribute(comfy_table::Attribute::Bold).set_alignment(comfy_table::CellAlignment::Center)).collect();
        table
        .load_preset(comfy_table::presets::UTF8_FULL)
        .apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS)
        .set_content_arrangement(comfy_table::ContentArrangement::Dynamic)
        .set_width(80)
        .set_header(header_cells);
    table.add_row(backends.values().map(|v| Cell::new(v.join("\n")).set_alignment(comfy_table::CellAlignment::Left)).collect::<Vec<_>>());
    println!("{table}");
    Ok(())
}
