use opencv::core::Mat;
use ort::Result as OrtResult;

pub trait SuperResolution {
    fn run(&mut self, input: Mat) -> OrtResult<Mat>;
}

pub mod realesrgan;