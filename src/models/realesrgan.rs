use crate::cancel::CancellationToken;
use crate::engine::{session_from_bytes, session_from_path};
use crate::models::SuperResolution;
use ndarray::{Array, Ix4};
use opencv::{core, core::Mat, dnn, imgproc, prelude::*};
use ort::{Result as OrtResult, session::Session, value::Tensor};
use std::path::Path;

/// Tiling configuration to bound memory and compute for large inputs.
pub struct TilingConfig {
    pub enabled: bool,
    pub tile: i32,
    pub pad: i32,
    pub threshold_pixels: i64,
    pub threshold_max_dim: i32,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tile: 512,
            pad: 16,
            threshold_pixels: 1_000_000,
            threshold_max_dim: 1024,
        }
    }
}

/// RealESRGAN model wrapper providing tiled and raw inference.
pub struct RealEsrgan {
    session: Session,
    tiling: TilingConfig,
}

impl RealEsrgan {
    pub fn from_path(path: &Path) -> OrtResult<Self> {
        let session = session_from_path(path)?;
        Ok(Self {
            session,
            tiling: TilingConfig::default(),
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> OrtResult<Self> {
        let session = session_from_bytes(bytes)?;
        Ok(Self {
            session,
            tiling: TilingConfig::default(),
        })
    }

    pub fn set_tiling_config(&mut self, cfg: TilingConfig) {
        self.tiling = cfg;
    }

    fn scale_from_depth(depth: i32) -> f64 {
        match depth {
            core::CV_8U => 1.0 / 255.0,
            core::CV_16U | core::CV_16S => 1.0 / 65535.0,
            _ => 1.0 / 255.0,
        }
    }

    fn ensure_bgr(input: &Mat, channels: i32) -> Result<Mat, String> {
        match channels {
            4 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_BGRA2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
                Ok(bgr)
            }
            1 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).map_err(|e| e.to_string())?;
                Ok(bgr)
            }
            _ => Ok(input.clone()),
        }
    }

    /// Ensure input size is valid for model reshape by padding to even width/height.
    fn pad_even(input: &Mat) -> Result<(Mat, i32, i32), String> {
        let w = input.cols();
        let h = input.rows();
        let pad_right = if w % 2 != 0 { 1 } else { 0 };
        let pad_bottom = if h % 2 != 0 { 1 } else { 0 };
        if pad_right == 0 && pad_bottom == 0 {
            return Ok((input.clone(), 0, 0));
        }
        let mut padded = Mat::default();
        // Use core::copy_make_border with REFLECT_101 to avoid visible seams
        core::copy_make_border(
            input,
            &mut padded,
            0,
            pad_bottom,
            0,
            pad_right,
            core::BORDER_REFLECT_101,
            core::Scalar::default(),
        )
        .map_err(|e| e.to_string())?;
        Ok((padded, pad_right, pad_bottom))
    }

    fn compose_bgra(out_bgr: &Mat, input: &Mat, depth: i32, w_out: i32, h_out: i32) -> Result<Mat, String> {
        let mut alpha = Mat::default();
        core::extract_channel(input, &mut alpha, 3).map_err(|e| e.to_string())?;

        let alpha_u8 = if depth == core::CV_8U {
            alpha
        } else {
            let mut tmp = Mat::default();
            alpha.convert_to(&mut tmp, core::CV_8U, 255.0 / 65535.0, 0.0).map_err(|e| e.to_string())?;
            tmp
        };

        let mut alpha_up = Mat::default();
        imgproc::resize(&alpha_u8, &mut alpha_up, core::Size::new(w_out, h_out), 0.0, 0.0, imgproc::INTER_LINEAR).map_err(|e| e.to_string())?;

        let mut out_rgba = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC4, core::Scalar::default()).map_err(|e| e.to_string())?;

        let mut src = core::Vector::<Mat>::new();
        src.push(out_bgr.clone());
        src.push(alpha_up);
        let from_to = [0, 0, 1, 1, 2, 2, 3, 3];
        core::mix_channels(&src, &mut out_rgba, &from_to).map_err(|e| e.to_string())?;
        Ok(out_rgba)
    }
}

impl SuperResolution for RealEsrgan {
    fn run(&mut self, input: Mat, token: &CancellationToken) -> Result<Mat, String> {
        if token.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let (h, w, channels, depth) = (input.rows(), input.cols(), input.channels(), input.depth());
        let bgr_input = Self::ensure_bgr(&input, channels)?;
        let (padded, pad_right, pad_bottom) = Self::pad_even(&bgr_input)?;
        let (w_pad, h_pad) = (padded.cols(), padded.rows());

        let use_tiling = self.needs_tiling(w_pad, h_pad);
        if token.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let (out_full, scale_est) = if use_tiling {
            self.infer_tiled(&padded, depth, token)?
        } else {
            self.infer_raw(&padded, depth, token)?
        };

        // Crop away padding to match original scaled size
        let target_w = w * scale_est;
        let target_h = h * scale_est;
        let mut out_cropped = out_full;
        if pad_right != 0 || pad_bottom != 0 {
            let roi = core::Rect::new(0, 0, target_w.max(1), target_h.max(1));
            let tmp = out_cropped.roi(roi).map_err(|e| e.to_string())?;
            out_cropped = tmp.try_clone().map_err(|e| e.to_string())?;
        }

        if channels == 4 {
            let out_rgba_full = Self::compose_bgra(&out_cropped, &input, depth, out_cropped.cols(), out_cropped.rows())?;
            Ok(out_rgba_full)
        } else {
            Ok(out_cropped)
        }
    }
}

impl RealEsrgan {
    fn needs_tiling(&self, w_pad: i32, h_pad: i32) -> bool {
        if !self.tiling.enabled {
            return false;
        }
        (w_pad as i64) * (h_pad as i64) > self.tiling.threshold_pixels || w_pad.max(h_pad) > self.tiling.threshold_max_dim
    }

    fn infer_raw(&mut self, padded_bgr: &Mat, depth: i32, token: &CancellationToken) -> Result<(Mat, i32), String> {
        if token.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let (h_pad, w_pad) = (padded_bgr.rows(), padded_bgr.cols());
        let scale = RealEsrgan::scale_from_depth(depth);
        let blob = dnn::blob_from_image(
            padded_bgr,
            scale,
            core::Size::new(w_pad, h_pad),
            core::Scalar::default(),
            true,
            false,
            core::CV_32F,
        )
        .map_err(|e| e.to_string())?;

        let blob_slice: &[f32] = blob.data_typed().map_err(|e| e.to_string())?;
        let input_array = Array::from_shape_vec((1, 3, h_pad as usize, w_pad as usize), blob_slice.to_vec()).map_err(|e| e.to_string())?;
        let input_tensor = Tensor::from_array(input_array).map_err(|e| e.to_string())?;

        if token.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let outputs = self.session.run(ort::inputs![input_tensor]).map_err(|e| e.to_string())?;
        let out_view = outputs[0].try_extract_array::<f32>().map_err(|e| e.to_string())?;
        let out4 = out_view.into_dimensionality::<Ix4>().map_err(|e| e.to_string())?;
        let (h_out, w_out) = (out4.shape()[2] as i32, out4.shape()[3] as i32);

        let mut out_mat = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC3, core::Scalar::default()).map_err(|e| e.to_string())?;
        let buf = out_mat.data_bytes_mut().map_err(|e| e.to_string())?;
        let w_out_usize = w_out as usize;
        for (y, row) in buf.chunks_mut(w_out_usize * 3).enumerate() {
            if token.is_cancelled() {
                return Err("Cancelled".to_string());
            }
            for x in 0..w_out_usize {
                let r = out4[[0, 0, y, x]].clamp(0.0, 1.0);
                let g = out4[[0, 1, y, x]].clamp(0.0, 1.0);
                let b = out4[[0, 2, y, x]].clamp(0.0, 1.0);
                let idx = x * 3;
                row[idx] = (b * 255.0).round() as u8;
                row[idx + 1] = (g * 255.0).round() as u8;
                row[idx + 2] = (r * 255.0).round() as u8;
            }
        }

        let scale_est_w = (w_out as f64) / (w_pad as f64);
        let scale_est_h = (h_out as f64) / (h_pad as f64);
        let scale_est = ((scale_est_w + scale_est_h) / 2.0).round() as i32;
        Ok((out_mat, scale_est))
    }

    fn infer_tiled(&mut self, input: &Mat, depth: i32, token: &CancellationToken) -> Result<(Mat, i32), String> {
        let (h, w) = (input.rows(), input.cols());
        let mut first_scale: Option<i32> = None;
        let mut out_canvas: Option<Mat> = None;

        let tile = self.tiling.tile;
        let tile_pad = self.tiling.pad;

        let mut y = 0;
        while y < h {
            if token.is_cancelled() {
                return Err("Cancelled".to_string());
            }
            let y0 = y;
            let y1 = (y0 + tile).min(h);
            let y0p = (y0 - tile_pad).max(0);
            let y1p = (y1 + tile_pad).min(h);
            let th = y1 - y0;

            let mut x = 0;
            while x < w {
                if token.is_cancelled() {
                    return Err("Cancelled".to_string());
                }
                let x0 = x;
                let x1 = (x0 + tile).min(w);
                let x0p = (x0 - tile_pad).max(0);
                let x1p = (x1 + tile_pad).min(w);
                let tw = x1 - x0;

                let roi_padded = core::Rect::new(x0p, y0p, (x1p - x0p).max(1), (y1p - y0p).max(1));
                let tile_padded = input.roi(roi_padded).map_err(|e| e.to_string())?.try_clone().map_err(|e| e.to_string())?;

                let (tile_even, _, _) = RealEsrgan::pad_even(&tile_padded)?;
                if token.is_cancelled() {
                    return Err("Cancelled".to_string());
                }
                let (tile_out, scale_est) = self.infer_raw(&tile_even, depth, token)?;

                if first_scale.is_none() {
                    first_scale = Some(scale_est);
                }
                let s = first_scale.unwrap();

                if out_canvas.is_none() {
                    let canvas = Mat::new_rows_cols_with_default(h * s, w * s, core::CV_8UC3, core::Scalar::default()).map_err(|e| e.to_string())?;
                    out_canvas = Some(canvas);
                }

                let crop_x = (x0 - x0p) * s;
                let crop_y = (y0 - y0p) * s;
                let crop_w = (tw.max(1)) * s;
                let crop_h = (th.max(1)) * s;
                let crop_rect = core::Rect::new(crop_x.max(0), crop_y.max(0), crop_w.max(1), crop_h.max(1));
                let tile_cropped = tile_out.roi(crop_rect).map_err(|e| e.to_string())?.try_clone().map_err(|e| e.to_string())?;

                let dst_x = x0 * s;
                let dst_y = y0 * s;
                let dst_roi = core::Rect::new(dst_x, dst_y, tile_cropped.cols(), tile_cropped.rows());
                let canvas_mut = out_canvas.as_mut().ok_or_else(|| "internal: canvas missing".to_string())?;
                let mut dst_view = canvas_mut.roi_mut(dst_roi).map_err(|e| e.to_string())?;
                tile_cropped.copy_to(&mut dst_view).map_err(|e| e.to_string())?;

                x += tile;
            }
            y += tile;
        }

        Ok((
            out_canvas.ok_or_else(|| "internal: canvas missing".to_string())?,
            first_scale.unwrap_or(4),
        ))
    }
}
