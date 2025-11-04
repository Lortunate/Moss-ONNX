use crate::engine::{session_from_bytes, session_from_path};
use crate::models::SuperResolution;
use ndarray::{Array, Ix4};
use opencv::{core, core::Mat, dnn, imgproc, prelude::*};
use ort::{Result as OrtResult, session::Session, value::Tensor};
use rayon::prelude::*;
use std::path::Path;

pub struct RealEsrgan {
    session: Session,
}

impl RealEsrgan {
    pub fn from_path(path: &Path) -> OrtResult<Self> {
        let session = session_from_path(path)?;
        Ok(Self { session })
    }

    pub fn from_bytes(bytes: &[u8]) -> OrtResult<Self> {
        let session = session_from_bytes(bytes)?;
        Ok(Self { session })
    }

    fn scale_from_depth(depth: i32) -> f64 {
        match depth {
            core::CV_8U => 1.0 / 255.0,
            core::CV_16U | core::CV_16S => 1.0 / 65535.0,
            _ => 1.0 / 255.0,
        }
    }

    fn ensure_bgr(input: &Mat, channels: i32) -> Mat {
        match channels {
            4 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_BGRA2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).unwrap();
                bgr
            }
            1 => {
                let mut bgr = Mat::default();
                imgproc::cvt_color(input, &mut bgr, imgproc::COLOR_GRAY2BGR, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT).unwrap();
                bgr
            }
            _ => input.clone(),
        }
    }

    fn compose_bgra(out_bgr: &Mat, input: &Mat, depth: i32, w_out: i32, h_out: i32) -> Mat {
        let mut alpha = Mat::default();
        core::extract_channel(input, &mut alpha, 3).unwrap();

        let alpha_u8 = if depth == core::CV_8U {
            alpha
        } else {
            let mut tmp = Mat::default();
            alpha.convert_to(&mut tmp, core::CV_8U, 255.0 / 65535.0, 0.0).unwrap();
            tmp
        };

        let mut alpha_up = Mat::default();
        imgproc::resize(&alpha_u8, &mut alpha_up, core::Size::new(w_out, h_out), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();

        let mut out_rgba = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC4, core::Scalar::default()).unwrap();

        let mut src = core::Vector::<Mat>::new();
        src.push(out_bgr.clone());
        src.push(alpha_up);
        let from_to = [0, 0, 1, 1, 2, 2, 3, 3];
        core::mix_channels(&src, &mut out_rgba, &from_to).unwrap();
        out_rgba
    }
}

impl SuperResolution for RealEsrgan {
    fn run(&mut self, input: Mat) -> OrtResult<Mat> {
        let (h, w, channels, depth) = (input.rows(), input.cols(), input.channels(), input.depth());
        let scale = Self::scale_from_depth(depth);
        let bgr_input = Self::ensure_bgr(&input, channels);

        let blob = dnn::blob_from_image(
            &bgr_input,
            scale,
            core::Size::new(w, h),
            core::Scalar::default(),
            true,
            false,
            core::CV_32F,
        )
        .unwrap();

        let blob_slice: &[f32] = blob.data_typed().unwrap();
        let input_array = Array::from_shape_vec((1, 3, h as usize, w as usize), blob_slice.to_vec()).unwrap();
        let input_tensor = Tensor::from_array(input_array)?;

        let outputs = self.session.run(ort::inputs![input_tensor])?;
        let out_view = outputs[0].try_extract_array::<f32>().unwrap();
        let out4 = out_view.into_dimensionality::<Ix4>().unwrap();
        let (h_out, w_out) = (out4.shape()[2] as i32, out4.shape()[3] as i32);

        let mut out_mat = Mat::new_rows_cols_with_default(h_out, w_out, core::CV_8UC3, core::Scalar::default()).unwrap();

        let buf = out_mat.data_bytes_mut().unwrap();
        let w_out_usize = w_out as usize;
        buf.par_chunks_mut(w_out_usize * 3).enumerate().for_each(|(y, row)| {
            for x in 0..w_out_usize {
                let r = out4[[0, 0, y, x]].clamp(0.0, 1.0);
                let g = out4[[0, 1, y, x]].clamp(0.0, 1.0);
                let b = out4[[0, 2, y, x]].clamp(0.0, 1.0);

                let idx = x * 3;
                row[idx] = (b * 255.0).round() as u8;
                row[idx + 1] = (g * 255.0).round() as u8;
                row[idx + 2] = (r * 255.0).round() as u8;
            }
        });

        if channels == 4 {
            Ok(Self::compose_bgra(&out_mat, &input, depth, w_out, h_out))
        } else {
            Ok(out_mat)
        }
    }
}
