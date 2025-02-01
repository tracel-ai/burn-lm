fn main() {
    // Define the default backend if there is not already a backend feature defined
    let has_features = std::env::vars().any(|(key, _)| key.starts_with("CARGO_FEATURE_"));
    if !has_features {
        println!("cargo:rustc-cfg=feature=\"wgpu\"");
    }
}
