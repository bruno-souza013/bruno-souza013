# CalmWave U-Net — Model Card Draft

> **Status: draft only. Do not upload this file as a final model card or publish model weights until the team has confirmed permission to redistribute the trained artifact, the dataset terms, and the applicable model license.** Replace all pending fields with verified information before publication.

## Model summary

CalmWave is a speech-enhancement model developed as part of a collaborative academic project. The model uses a convolutional U-Net adapted to the magnitude spectrogram domain to estimate a mask that attenuates background noise while aiming to preserve speech characteristics.

## Intended use

Research and educational experimentation with speech enhancement and on-device inference. The model is not a hearing device, diagnostic tool, clinical system, or substitute for professional evaluation.

## Architecture

- Model family: convolutional U-Net
- Framework used for training: PyTorch
- Deployment artifact described in the paper: ONNX
- Runtime target: Android with ONNX Runtime
- Approximate model size: 7.7 million parameters

## Training data and procedure

The accompanying paper reports use of the VoiceBank-DEMAND dataset, resampled to 16 kHz, with 80% training, 10% validation and 10% test splits. It describes 100 training epochs, batch size 16, Adam optimizer and L1/MAE loss.

Before publication, confirm the exact dataset acquisition source and license/terms, and make sure they permit the proposed redistribution and intended use.

## Evaluation reported in the paper

The paper presents results from more than one evaluation context. Do not combine them as if they came from the same run.

| Evaluation context | PESQ | STOI | SDR |
|---|---:|---:|---:|
| Reported model evaluation | 2.445 | 0.860 | 14.98 dB |
| Average across tested smartphones | 2.03 | 0.89 | 11.94 dB |

The paper also reports an approximate end-to-end latency of 200 ms. Results may vary by hardware and audio conditions. Include a link to the paper and the exact evaluation script/commit when those artifacts are ready.

## Limitations and risks

- The paper's qualitative evaluation included only two participants and is a preliminary proof of concept, not a formal clinical validation.
- Overlapping speech from multiple simultaneous speakers may be more challenging than continuous background noise.
- Latency, battery usage and output quality may vary by smartphone hardware and acoustic conditions.
- The implementation described in the paper targets Android.

## Team and attribution

This model was developed as part of a team project. Confirm the preferred citation and author list with the coauthors. Do not describe team-level training or evaluation as an individual contribution unless that accurately reflects your role.

## Pending before publishing

- [ ] Obtain explicit approval from project collaborators to publish the trained weights.
- [ ] Verify training-dataset terms, distribution rights and attribution requirements.
- [ ] Choose and document the model license; do not assume a license.
- [ ] Upload only the authorized inference artifact and relevant configuration.
- [ ] Add an exact paper link, source-code link, version/commit and evaluation procedure.
- [ ] Test the upload and inference instructions from a clean environment.

## Suggested metadata after permissions are confirmed

Add a YAML metadata block at the top of this file only after verifying task/library tags and choosing an accurate license. Hugging Face Model Cards are README files with metadata plus human-readable details about intended use, training, evaluation, limitations and licensing.
