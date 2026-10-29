# CalmWave — AI Speech Enhancement on Edge

## Project overview
CalmWave is a team-built Android application that applies neural speech enhancement to reduce background noise and make speech clearer in acoustically complex environments.

## AI approach
The project uses a convolutional U-Net adapted to the audio spectrogram domain. The paper describes training with PyTorch on VoiceBank-DEMAND at 16 kHz, with an 80/10/10 train/validation/test split, for 100 epochs using Adam and L1/MAE loss. The network has approximately 7.7 million parameters and estimates a multiplicative mask over the magnitude spectrogram.

## Inference pipeline
1. Capture PCM audio.
2. Compute a Short-Time Fourier Transform (STFT).
3. Apply logarithmic magnitude compression.
4. Run the U-Net to estimate a spectral mask.
5. Apply the mask to attenuate noise while retaining speech characteristics.
6. Reconstruct the output with inverse STFT (ISTFT).

## On-device integration
The trained model is exported from PyTorch to ONNX and executed on Android using ONNX Runtime. The architecture uses the Android NNAPI where available, with optimized CPU execution as a fallback. The audio filtering pipeline runs locally rather than uploading raw voice audio to a remote inference service.

**Related technologies:** Python, PyTorch, U-Net, ONNX, ONNX Runtime, Kotlin, Android, Jetpack Compose.

## Results reported in the project paper
The following numbers represent distinct evaluation contexts and should not be treated as measurements from one identical run.

| Evaluation | PESQ | STOI | SDR |
|---|---:|---:|---:|
| Reported model evaluation | 2.445 | 0.860 | 14.98 dB |
| Average across tested smartphones | 2.03 | 0.89 | 11.94 dB |

The paper reports an end-to-end latency of approximately 200 ms and local inference time around 150 ms. Results can vary with device, audio sample and evaluation conditions.

## Limitations
- The qualitative user evaluation was a small preliminary proof of concept, not a formal clinical validation.
- Simultaneous overlapping voices can be more difficult to separate than continuous background noise.
- Performance and battery usage vary by smartphone hardware.
- The implementation described in the paper targets Android.

## Team contribution
This was a collaborative academic project. My individual contribution focused on the mobile application and audio/user interaction flows. Model training and reported evaluation results are attributed to the team project as documented in the paper, not solely to one contributor.

## Reproducibility and artifacts
Before redistributing model weights, datasets or code from the team repository, confirm the licenses and obtain authorization from the coauthors. This repository page documents the technical approach; it does not itself redistribute the trained model.
