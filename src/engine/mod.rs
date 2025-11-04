use std::path::Path;

use ort::Result as OrtResult;
use ort::execution_providers::*;
use ort::session::Session;
use ort::session::builder::SessionBuilder;

fn create_common_session_builder() -> OrtResult<SessionBuilder> {
    let num_threads = num_cpus::get().max(1);
    let mut providers: Vec<ExecutionProviderDispatch> = Vec::new();

    #[cfg(target_os = "android")]
    {
        providers.push(NNAPIExecutionProvider::default().build());
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        providers.push(CoreMLExecutionProvider::default().build());
    }

    #[cfg(target_os = "windows")]
    {
        providers.push(CUDAExecutionProvider::default().build());
        #[cfg(feature = "directml")]
        {
            providers.push(DirectMLExecutionProvider::default().build());
        }
    }

    #[cfg(target_os = "linux")]
    {
        providers.push(CUDAExecutionProvider::default().build());
        #[cfg(feature = "tensorrt")]
        {
            providers.push(TensorRTExecutionProvider::default().build());
        }
    }
    providers.push(CPUExecutionProvider::default().build());

    let builder = Session::builder()?
        .with_inter_threads(num_threads)?
        .with_intra_threads(num_threads)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
        .with_execution_providers(providers)?;
    Ok(builder)
}

pub fn session_from_path(path: &Path) -> OrtResult<Session> {
    create_common_session_builder()?.commit_from_file(path)
}

pub fn session_from_bytes(bytes: &[u8]) -> OrtResult<Session> {
    create_common_session_builder()?.commit_from_memory(bytes)
}
