use crate::cancel::CancellationToken;
use crate::models::SuperResolution;
use opencv::{core, imgproc, prelude::*};

#[derive(Clone, Copy)]
pub enum ScaleStrategy {
    /// One SR pass at base_scale, then interpolate to target.
    SinglePass,
    /// SR passes so that base^passes is nearest to target (minimize final interpolation).
    NearestPower,
    /// SR passes so that base^passes >= target, favor slight downscale at the end.
    CeilPower,
    /// SR passes so that base^passes <= target, favor slight upscale at the end.
    FloorPower,
}

pub struct SrPipeline {
    model: Box<dyn SuperResolution>,
    base_scale: f64,
    strategy: ScaleStrategy,
}

impl SrPipeline {
    pub fn new(model: Box<dyn SuperResolution>, base_scale: f64) -> Self {
        // Default: NearestPower to minimize final interpolation from achieved scale to target.
        Self {
            model,
            base_scale,
            strategy: ScaleStrategy::NearestPower,
        }
    }

    pub fn set_strategy(&mut self, strategy: ScaleStrategy) {
        self.strategy = strategy;
    }

    pub fn run_to_scale(&mut self, input: Mat, target_scale: f64, token: &CancellationToken) -> Result<Mat, String> {
        // Strategy selects how many SR passes to run before final interpolation.
        let passes = match self.strategy {
            ScaleStrategy::SinglePass => 1,
            ScaleStrategy::NearestPower => {
                let n_real = target_scale.ln() / self.base_scale.ln();
                n_real.round() as i32
            }
            ScaleStrategy::CeilPower => {
                let n_real = target_scale.ln() / self.base_scale.ln();
                n_real.ceil() as i32
            }
            ScaleStrategy::FloorPower => {
                let n_real = target_scale.ln() / self.base_scale.ln();
                n_real.floor() as i32
            }
        }
        .max(0);

        // Apply SR iteratively.
        let mut current = input;
        for _ in 0..passes {
            if token.is_cancelled() {
                return Err("Cancelled".to_string());
            }
            current = self.model.run(current, token)?;
        }

        // Ratio from achieved scale to target.
        let achieved_scale = if passes == 0 { 1.0 } else { self.base_scale.powi(passes) };
        let ratio = target_scale / achieved_scale;

        if (ratio - 1.0).abs() < f64::EPSILON {
            return Ok(current);
        }

        if token.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let w_cur = current.cols();
        let h_cur = current.rows();
        let w_target = ((w_cur as f64) * ratio).round() as i32;
        let h_target = ((h_cur as f64) * ratio).round() as i32;

        let mut out = Mat::default();
        let interp = if ratio < 1.0 { imgproc::INTER_AREA } else { imgproc::INTER_LANCZOS4 };
        imgproc::resize(&current, &mut out, core::Size::new(w_target, h_target), 0.0, 0.0, interp).map_err(|e| e.to_string())?;

        Ok(out)
    }
}
