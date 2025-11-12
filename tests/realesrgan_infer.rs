#[test]
fn infer_realesrgan() {
    use moss_model::{CancellationToken, RealEsrgan, SuperResolution};
    use opencv::imgcodecs;
    use std::fs;
    use std::path::Path;

    let model_path = Path::new("models/x4plus.onnx");
    let mut sr = RealEsrgan::from_path(model_path).unwrap();
    let token = CancellationToken::new();

    let _ = fs::create_dir_all("tests/results");

    let img_dir = Path::new("tests/imgs");
    let entries = fs::read_dir(img_dir).unwrap();
    let exts = ["jpg", "png"];

    let mut processed = 0;
    for entry in entries {
        let entry = entry.unwrap();
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
            Some(e) if exts.contains(&e.as_str()) => e,
            _ => continue,
        };

        let input = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
        let start = std::time::Instant::now();
        let output = sr.run(input, &token).unwrap();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("{}: {:.2}ms", stem, elapsed_ms);
        let out_path = format!("tests/results/{}_x4.{}", stem, ext);

        imgcodecs::imwrite(&out_path, &output, &opencv::core::Vector::new()).unwrap();
        assert!(Path::new(&out_path).exists());
        processed += 1;
    }

    assert!(processed > 0);
}
