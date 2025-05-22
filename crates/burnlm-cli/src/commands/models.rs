use burnlm_registry::Registry;
use comfy_table::{Cell, Table};

pub(crate) fn create() -> clap::Command {
    clap::Command::new("models").about("List all available models and their download status")
}

pub(crate) fn handle(shell: bool) -> super::HandleCommandResult {
    let registry = Registry::new();
    let mut table = Table::new();
    table
        .load_preset(comfy_table::presets::UTF8_FULL)
        .apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS)
        .set_content_arrangement(comfy_table::ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Available Models")
                .add_attribute(comfy_table::Attribute::Bold)
                .set_alignment(comfy_table::CellAlignment::Center),
            Cell::new("Downloaded")
                .add_attribute(comfy_table::Attribute::Bold)
                .set_alignment(comfy_table::CellAlignment::Center),
            Cell::new("Download Command")
                .add_attribute(comfy_table::Attribute::Bold)
                .set_alignment(comfy_table::CellAlignment::Center),
        ]);
    let mut reg_entries: Vec<_> = registry.get().iter().collect();
    reg_entries.sort_by_key(|(key, ..)| *key);
    for (name, plugin) in reg_entries {
        let installation_status = if plugin.is_downloaded() { "✅" } else { "❌" };
        let install_cmd_cell = if plugin.downloader().is_some() {
            let content = format!(
                "{}download {}",
                if shell { "" } else { "cargo burnlm " },
                plugin.model_cli_param_name()
            );
            Cell::new(content).set_alignment(comfy_table::CellAlignment::Left)
        } else {
            Cell::new("─").set_alignment(comfy_table::CellAlignment::Center)
        };
        table.add_row(vec![
            Cell::new(name).set_alignment(comfy_table::CellAlignment::Left),
            Cell::new(installation_status).set_alignment(comfy_table::CellAlignment::Center),
            install_cmd_cell,
        ]);
    }
    println!("{table}");
    Ok(None)
}
