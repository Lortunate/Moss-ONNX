use crate::cancel::CancellationToken;
use opencv::core::Mat;

pub trait SuperResolution {
    fn run(&mut self, input: Mat, token: &CancellationToken) -> Result<Mat, String>;
}

pub mod realesrgan;
