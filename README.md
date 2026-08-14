# ACE-Step C++ Cog

Portable text-to-music generation using [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), the native [acestep.cpp](https://github.com/ace-step/acestep.cpp) GGML runtime, and the standard [Cog](https://github.com/replicate/cog) prediction contract.

This repository contains no model weights. On first setup it downloads the four GGUF files selected by `ACESTEP_QUANT` and `ACESTEP_LM_SIZE` into the persistent model directory. Generated audio is stereo 48 kHz MP3 or WAV.

## Inputs

| Input | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Style, mood, instruments, production, and scene |
| `lyrics` | `[Instrumental]` | Structured lyrics, blank for model-written lyrics, or `[Instrumental]` |
| `duration` | `30` | 10–300 seconds |
| `bpm` | `0` | 0 lets the model choose; otherwise 40–240 |
| `key` | blank | For example `D minor`; blank lets the model choose |
| `seed` | `17` | Deterministic seed |
| `steps` | `8` | Turbo generation uses eight steps |
| `format` | `mp3` | `mp3` or `wav` |

## Run

```bash
cog build -t ghcr.io/lee101/acestep-cpp-cog:local
cog predict \
  -i prompt="minimal felt piano, intimate room, 84 BPM" \
  -i lyrics="[Instrumental]" \
  -i duration=30 \
  -i bpm=84 \
  -i key="D minor"
```

The image starts `ace-server` inside `setup()`, submits the asynchronous LM and synthesis stages, polls each native job, extracts the audio part from the multipart result, and returns one Cog `Path`.

## Configuration

| Variable | Default |
| --- | --- |
| `ACESTEP_MODELS_DIR` | `/weights/acestep` |
| `ACESTEP_QUANT` | `Q8_0` |
| `ACESTEP_LM_SIZE` | `0.6B` |
| `ACESTEP_BACKEND` | `CUDA0` |
| `ACESTEP_SERVER_URL` | starts the bundled server |
| `ACESTEP_KEEP_LOADED` | `1` |
| `ACESTEP_JOB_TIMEOUT` | `900` seconds |
| `ACESTEP_ENGINE` | `native`; use `contract` only for weight-free CI |

Use a persistent volume for `/weights`. The 0.6B LM is the lower-latency serverless default; set `ACESTEP_LM_SIZE=4B` when quality matters more than cold-start size.

## app.nz

`appnz.schema.json` describes the same input contract for an app.nz Cog template. Deploy the built image as `music-diffusion-native`, keep minimum workers at zero, and mount the model cache at `/weights`.

## Licensing

This adapter is MIT-licensed. ACE-Step 1.5, acestep.cpp, and downloaded weights retain their own licenses and notices. MiniMax Music is not bundled: its hosted model has no public weights and is a separate proprietary provider.
