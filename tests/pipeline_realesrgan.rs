#[cfg(test)]
mod tests {
    use moss_model::pipeline::ScaleStrategy;
    use moss_model::{CancellationToken, RealEsrgan, SrPipeline};
    use opencv::{core, imgcodecs, prelude::*};
    use std::fs;
    use std::path::{Path, PathBuf};

    fn load_test_images() -> Vec<(PathBuf, Mat)> {
        let img_dir = Path::new("tests/imgs");
        let entries = fs::read_dir(img_dir).unwrap();
        let exts = ["jpg", "png"];
        let mut imgs = Vec::new();
        for entry in entries {
            let entry = entry.unwrap();
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            match path.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()) {
                Some(e) if exts.contains(&e.as_str()) => {}
                _ => continue,
            };
            let img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_UNCHANGED).unwrap();
            assert!(!img.empty());
            imgs.push((path, img));
        }
        assert!(!imgs.is_empty());
        imgs
    }

    fn dims(mat: &Mat) -> (i32, i32) {
        (mat.cols(), mat.rows())
    }

    fn expected_dims(w: i32, h: i32, target_scale: f64) -> (i32, i32) {
        let w_t = ((w as f64) * target_scale).round() as i32;
        let h_t = ((h as f64) * target_scale).round() as i32;
        (w_t, h_t)
    }

    fn strategy_name(s: ScaleStrategy) -> &'static str {
        match s {
            ScaleStrategy::SinglePass => "single",
            ScaleStrategy::NearestPower => "nearest",
            ScaleStrategy::CeilPower => "ceil",
            ScaleStrategy::FloorPower => "floor",
        }
    }

    fn scale_label(scale: f64) -> String {
        let i = scale.round() as i32;
        if (scale - i as f64).abs() < f64::EPSILON {
            format!("x{}", i)
        } else {
            format!("x{:.2}", scale)
        }
    }

    fn run_pipeline_over_images(model_path: &str, strategy: ScaleStrategy, target_scale: f64) {
        let model = RealEsrgan::from_path(Path::new(model_path)).unwrap();
        let base_scale = 4.0; // RealESRGAN x4 models
        let mut pipe = SrPipeline::new(Box::new(model), base_scale);
        pipe.set_strategy(strategy);
        let token = CancellationToken::new();

        let model_label = Path::new(model_path).file_stem().and_then(|s| s.to_str()).unwrap();
        let strat_label = strategy_name(strategy);
        let scale_lbl = scale_label(target_scale);
        let out_dir = Path::new("tests/results").join(model_label).join(strat_label);
        let _ = fs::create_dir_all(&out_dir);

        for (path, input) in load_test_images() {
            let (w_in, h_in) = dims(&input);
            let out = pipe.run_to_scale(input, target_scale, &token).unwrap();
            let (w_o, h_o) = dims(&out);
            let (w_e, h_e) = expected_dims(w_in, h_in, target_scale);
            assert_eq!((w_o, h_o), (w_e, h_e));

            // Save output image for inspection
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
            let ext = path.extension().and_then(|s| s.to_str()).unwrap();
            let out_path = out_dir.join(format!("{}_{}.{}", stem, scale_lbl, ext));
            imgcodecs::imwrite(out_path.to_str().unwrap(), &out, &core::Vector::new()).unwrap();
            assert!(out_dir.join(format!("{}_{}.{}", stem, scale_lbl, ext)).exists());
        }
    }

    #[test]
    fn pipeline_realesrgan_generic_single_pass() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::SinglePass, 3.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_nearest_power() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::NearestPower, 3.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_ceil_power() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::CeilPower, 3.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_floor_power() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::FloorPower, 3.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_single_pass_mult() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::SinglePass, 12.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_nearest_power_mult() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::NearestPower, 12.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_ceil_power_mult() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::CeilPower, 12.0);
    }

    #[test]
    fn pipeline_realesrgan_generic_floor_power_mult() {
        run_pipeline_over_images("models/x4plus.onnx", ScaleStrategy::FloorPower, 12.0);
    }
}
