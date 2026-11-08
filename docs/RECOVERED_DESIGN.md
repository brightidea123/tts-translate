# Design extracted from the saved source

This document describes [`tts-transcribe.py`](../tts-transcribe.py) at commit `65be3e4bef2133d5f1be309fe1af42e23c8f6071`. It is an analysis of saved code, not a transcript of earlier planning or proof that every implemented backend worked. Setup instructions and future reconstruction requirements are inferred from imports, argument definitions, and call sites.

## Product behavior

The application is a terminal-based duplex translator. Incoming source-language audio is transcribed in its original language, then translated into the target language. Typed target-language text follows a separate reverse path: translation into the source language, speech synthesis, and local playback. Defaults are Turkish source and English target.

Languages are Arabic, German, English, Spanish, French, Hebrew, Russian, and Turkish. Language metadata carries Whisper codes, NLLB codes, native scripts, locale names, and default voice names. Aliases include `SP`/`SPA` for Spanish, `ENG`, `TUR`, `ARA`/`ARB`, `GER`/`DEU`, `FRE`/`FRA`, `HEB`, and `RUS`. Source and target must be different.

`audio`, `text`, and `both` modes enable the incoming-audio path, typed-reply path, or both. Live execution requires a TTY. The dashboard displays incoming and typed-reply activity separately, including partial stability and in-flight quality markers. Enter sends typed text or flushes audio when no text is pending; Escape cancels input before exiting. Device routing remains external to the program: the source captures an input device and does not install a call-audio loopback route.

## Pipeline and concurrency

```text
input device / native bridge
  -> lightweight capture callback
  -> VAD + endpointing + bounded audio storage
  -> partial/final ASR scheduling
  -> transcript coordinator + cleanup
  -> incoming translation -> dashboard / optional incoming TTS

keyboard -> committed reverse translation -> reply TTS -> playback
```

The code separates capture, recognition, translation, and speech synthesis. Translation runs in a separate process by default (`--no-mt-process` disables this). The legacy audio engine uses separate ASR processing and result coordination; the native ring engine uses a bounded audio ring, endpoint detector, and priority ASR scheduler. Workers communicate through bounded queues, with shared-memory/slot helpers for audio in the legacy path.

Queue policies preserve final ASR work and typed replies while discarding stale speculative work. A new partial should not displace a queued final. A committed typed translation can displace a speculative incoming translation. Typed TTS is protected from stale optional incoming-audio TTS. `--partial-queue-latest-only` keeps the newest partial per utterance; final jobs preempt partials.

Audio is mono signed Int16 PCM at **16,000 Hz**, with **20 ms / 320-sample** frames. The capture callback delegates work rather than loading models or translating. Settings control capture buffering, overlapping recognition windows, maximum pending jobs, and how long the native ring retains audio.

## Endpointing, drafts, and final revisions

VAD can use energy detection, WebRTC, or Silero. Endpointing combines a noise gate, speech ratio, hangover/tail padding, adaptive silence thresholds, and optional semantic completeness from partial transcripts. Long speech is flushed into overlapping chunks. A short post-endpoint hold can defer finalization if speech resumes.

Partial source display can show stable text, complete clumps, or the latest speculative text (`--partial-display-policy`). Stable-prefix locking uses repeated agreement and retains a mutable tail. Partial translation policies are `stable`, `speculative`, and `off`; stabilization time, word count, and completeness gates control when drafts are useful enough to translate. Speculative translations are display-only. MLX translation can stream display deltas separately from committed output and chunked reply speech.

Final recognition may merge useful details from the last partial. Final translation completeness policies are `off`, `conservative`, and `aggressive`. Watchdogs close or finalize stuck partials, expire markers that never receive text, and handle empty finals using fallback, close, or keep policies. A final-ASR pending gate prevents premature watchdog fallback.

Cleanup handles repeated overlapping ASR fragments, recent near-duplicate finals across endpoint bounce, known subtitle-credit hallucinations, model wrappers/thought channels, and prompt echoes. Source-language forcing is on by default; strict script checks reject likely translation or romanization when native-script source text was expected. English-only Whisper models are guarded for non-English sources. ASR initial prompts default to empty to avoid prompt-language leakage.

Arabic handling normalizes presentation forms before processing and logging, while retaining canonical terminal text. Terminal width and lam-alef divider compensation are handled separately from text content.

## Profiles

Profile defaults are applied unless the corresponding positive or negative flag was supplied explicitly. They are configuration bundles, not benchmarked performance guarantees.

| Profile | ASR beam / MT beams | Streaming chunk / partial window | Base silence | Main intent in saved configuration |
| --- | --- | --- | --- | --- |
| `live` | 2 / 3 | 0.64 s / 1.8 s | 360 ms | Small queues and frequent partial updates; default |
| `balanced` | 3 / 4 | 1.1 s / 2.8 s | 440 ms | More buffering and context |
| `stable` | 2 / 4 | 1.0 s / 2.6 s | 500 ms | Stricter anti-repeat, hallucination, confidence, and stable-prefix gates |
| `accurate` | 5 / 5 | 3.8 s / 6.0 s | 700 ms | Longer context and larger queues |
| `m4-max-live` | 2 / 3 | 0.85 s / 2.2 s | 360 ms | Explicit MLX Whisper + MLX LM + Apple VPIO/ring route |

Every profile sets the Whisper model to `large-v3-turbo` unless overridden. Backend-specific defaults determine the actual translation model. The M4 profile sets separate partial/final Whisper model names, smaller queues, stable translation, and separate translation processing. Its original native helper was absent from the saved source; a reconstructed replacement is now provided separately, as described below.

## Backend selection

### Speech recognition

Main STT choices are `auto`, `mlx-vlm-gemma`, `mlx-whisper`, `dictate-stt`, `faster-whisper`, and `whisper-cli`. On Apple Silicon, auto selection prefers installed MLX VLM, then MLX Whisper, then the external dictation or whisper.cpp commands, then Faster Whisper. Other macOS selection prefers available whisper.cpp or dictation commands before Faster Whisper.

The MLX paths use Metal. The dictation path labels its automatic device `ane`; whisper.cpp labels it `metal`. These are source selection labels and do not establish actual hardware utilization by external commands. Faster Whisper is explicitly instantiated on CPU with the selected compute type.

Partial and final lanes support `same` or explicit backend overrides. The partial lane additionally supports `apple-speech`; the final lane supports `whispercpp-coreml`. Separate lane model names, beams, and Gemma prompts are configurable. MLX Whisper attempts beam decoding when supported and caches a greedy fallback when it is not.

MLX Whisper model names resolve to `mlx-community/whisper-<name>` unless a local path or repository reference is supplied. whisper.cpp needs a local GGML model supplied by path or found under `~/.cache/whisper`, `~/.cache/whisper.cpp`, or `~/.local/share/whisper`. The saved default Gemma reference is `mlx-community/gemma-4-e4b-it-4bit`; its availability and compatibility were not verified during restoration.

### Translation

Choices are `auto`, `mlx-vlm-gemma`, `mlx-lm`, and `nllb`. Auto uses Gemma translation when the main STT flag explicitly selects Gemma and MLX VLM is installed; otherwise Apple Silicon with MLX LM installed uses MLX LM, then falls back to NLLB. Automatic MLX initialization failures can fall back further; an explicitly requested unavailable backend raises an error.

Saved model defaults are `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` for MLX LM, the Gemma audio reference for MLX VLM, and `facebook/nllb-200-distilled-600M` for NLLB. NLLB uses PyTorch with MPS when available. Its Hugging Face loading tries local files before allowing a download. MLX loaders can also fetch model assets.

Custom MT prompts accept source language, target language, and text placeholders. Gemma ASR prompts accept language codes/names and task hints; the default asks for source-language transcription rather than translation. Model references are saved configuration, not recovered model weights.

### Audio and speech detection

Audio choices are `auto`, `sounddevice`, `apple-raw`, and `apple-vpio`. Auto selects the adjacent native helper on Apple Silicon when that path exists, otherwise SoundDevice. The ring engine requires a native audio backend and helper; the legacy engine also works with SoundDevice. Explicit `--audio-io sounddevice --streaming-engine legacy` avoids the missing helper.

VAD auto chooses installed Silero with PyTorch, then WebRTC, then energy-only detection. Silero can use ONNX or PyTorch, preferring ONNX when ONNX Runtime is installed; the saved integration still requires PyTorch even for Silero ONNX mode. Duplex capture, TTS bleed ducking, playback guard time, VPIO AGC, and processing bypass are configurable.

### Speech synthesis and playback

TTS choices are `auto`, `kokoro`, `say`, `piper`, and `edge`. Automatic preference varies by language: English tries Kokoro first; Turkish tries Piper first; other languages try Edge first. The implementation attempts further choices when synthesis fails. Explicit TTS selection limits the route.

Kokoro expects `~/models/kokoro/kokoro-v1.0.onnx` and `~/models/kokoro/voices-v1.0.bin` unless overridden. Piper expects `~/.local/share/piper/tr_TR-fahrettin-medium.onnx` and the compatible CLI syntax `piper -m <model> -f <wav>`. This Piper path is Turkish-specific. macOS speech uses installed voices plus `say` and `afconvert`; missing voices can result in a fallback voice. Edge calls `edge-tts`, requires network access, and converts generated MP3 through `ffmpeg` or macOS `afconvert`.

Native helper playback uses its stdin protocol below. Other local playback uses `afplay` on macOS, otherwise `ffplay` if present. Reply TTS can speak translated chunks as they complete. Speaking incoming-audio translations is optional and disabled by default.

## Missing native helper contract

The original `apple_vpio_bridge.swift` and its compiled `apple_vpio_bridge` were missing from the saved repository. The original timestamped capture source has since been recovered, and a raw-duplex replacement has been reconstructed as a clearly identified derivative. See [`NATIVE_BRIDGE_RECOVERY.md`](NATIVE_BRIDGE_RECOVERY.md) for hashes, build/check commands, and validation limits. The protocol below is still extracted from Python, not from a recovered original raw helper. The Python dependency error contains this build hint:

```sh
swiftc apple_vpio_bridge.swift -framework AVFoundation -o apple_vpio_bridge
```

This can now build the reconstructed derivative. `--apple-bridge-path` selects another binary; the default path is beside the Python entry point. The active `AppleVPIOBridge` wrapper expects:

| Channel or setting | Contract extracted from Python |
| --- | --- |
| Child stdout | Unframed 16 kHz mono signed Int16 PCM; Python accumulates 640 bytes and dispatches each 320-sample / 20 ms frame |
| Child stdin | Repeated playback messages: four-byte unsigned little-endian byte length, followed by that many bytes containing a complete WAV file |
| Child stderr readiness | A line beginning `apple-vpio-bridge ready`; an optional `mode=` suffix is displayed |
| Child stderr no input | Exact line `no input frames received within 2s` signals stalled capture |
| Other stderr | Diagnostic lines; recognized verbose tap/input/output/playback/channel lines are treated as debug output |
| `APPLE_VPIO_MODE` | `vpio` or `raw` |
| `APPLE_VPIO_AGC` | `on` or `off`, from `--apple-vpio-agc` |
| `APPLE_VPIO_BYPASS` | `on` or `off`, from `--apple-vpio-bypass-processing` |
| `APPLE_VPIO_VERBOSE` | Set to `1` for verbose or debug sessions |
| `APPLE_VPIO_DEBUG` | Set to `1` for debug sessions |

The wrapper waits up to three seconds for readiness and observes the no-input event. For VPIO mode, a no-input event with no captured chunks can trigger a retry in raw mode. Stopping a VPIO process includes a three-second audio-route cooldown. Both the main ring route and the legacy native route currently use this raw-PCM bridge; ring timestamps are introduced by the Python feeder.

A separate `TimestampedAppleVPIOBridge` class exists but is not used by the main ring startup. It describes a different stdout format: a 20-byte header containing `APCM`, then little-endian `<QII>` fields for timestamp in nanoseconds, sample rate, and sample count, followed by `sample_count * 2` PCM bytes. It has no playback stdin. A replacement helper for the active entry point must implement the raw-PCM duplex contract above rather than selecting this unused protocol by its class name.

## Other external helper requirements

Apple Speech partial recognition requires an explicit `--apple-speech-command`; the saved repository does not provide that command. Python writes a temporary 16 kHz mono Int16 WAV and invokes the selected executable as `<command> <wav_path> <source_locale>`, for example with locale `tr-TR`. It expects exit status zero and stdout containing either transcript text or a JSON object with a `text` field. Captured stderr is not interpreted as the transcript.

whisper.cpp/Core ML finals require `--whispercpp-command` and a local `--whispercpp-model`. Python supplies `-m <model>`, optional forced-language `-l <code>`, `-f <wav>`, `-t <threads>`, `-bs <beam>`, `-bo 1`, `-np`, and `-nt`, followed by an optional `--prompt` and shell-split `--whispercpp-extra-args`. The process is invoked without a shell and must exit successfully. Transcript extraction combines stdout/stderr and filters known whisper.cpp diagnostic prefixes. The backend's name does not build or enable Core ML by itself; the external binary/model configuration must provide that capability.

Other CLI STT backends also depend on command interfaces implemented outside this repository. No Apple Speech helper, whisper.cpp build, dictation binary, Core ML model assets, voice files, loopback setup, or model caches were recovered.

## Runtime dependencies and reconstructed setup

Python 3.10+ is required by dataclass slots; Python 3.12 is the recommended setup target. NumPy is unconditional. SoundDevice, WebRTC, PyTorch, Faster Whisper, Transformers, and Kokoro imports are optional; MLX packages and Silero/ONNX availability are detected before later loading. A broken installed optional package can still fail at import time if its failure is not an `ImportError`.

For the helper-free Apple Silicon route, install NumPy, MLX Whisper, MLX LM, and SoundDevice; choose energy VAD and macOS `say`. Optional alternatives and reconstructed installation commands are listed in the README. No lockfile or original dependency versions were recovered, and optional installation/model inference has not been validated in this offline restoration environment. Live startup may download models; use diagnostic commands first.

## Validation and remaining limits

The existing `--self-test` path runs before device/model initialization. Its 23 checks cover queue priorities, final-pending gates, cached MLX beam fallback, language forcing, Gemma prompt/output sanitation, hallucination/repeat cleanup, duplicate finals, complete-clump translation, stable/draft UI behavior, stale-marker expiry, MLX context preservation, prompt text insertion, final-detail merging, and Arabic normalization/rendering.

All 23 checks passed during restoration using an available bundled Python 3.12 runtime with NumPy. They use synthetic inputs or stubs and do not validate model downloads, speech accuracy, latency, hardware scheduling, device permissions, or native bridge behavior. `--help` and `--dump-config` also exit without live model/audio initialization. `--list-devices` does inspect the selected audio runtime and is not the same kind of offline check.

The source appends finalized transcript/translation records to `transcription.log` and typed-reply records to `tts.log`, unless `--no-session-logs` is supplied. Final transcription logging is debounced so refinements can replace earlier text before writing. Generated reply audio defaults to `output.wav`; optional `--save-last-audio` adds diagnostic capture output. Runtime data and any prior personalized configuration were not recovered.

## November 6 implementation updates

The preceding source-derived design describes the recovered baseline. Newly
implemented changes allow `--streaming-engine ring --audio-io sounddevice`,
including a selected LitLink input. A bounded `RingAudioFeeder` assembles fixed
PCM frames, assigns sequence numbers, maps PortAudio ADC time to the monotonic
clock, and feeds endpoint/VAD processing from a separate worker. Native raw
Apple capture uses the same ingress, with sample timing anchored to receipt
because its stream has no hardware timestamps. One source is selected per
process; independent capture clocks are not merged.

Ingress drops and device overruns produce reported discontinuities and new
utterance boundaries. Ring retention is sample-exact across wraps and oversized
writes. Partial queue eviction is atomic and preserves submitted final work;
late partial results cannot republish a closed utterance or clear newer partial
ownership. TranslateGemma uses structured original text and source/target codes,
while existing generic/custom translation prompts retain their prior behavior.

The restored environment now uses Python 3.12 and exact current compatible MLX
package pins. Model assets, actual checks and remaining Metal/audio restrictions
are recorded in [runtime validation](RUNTIME_VALIDATION.md). These changes are
new reconstruction work, not recovered October implementation. Immediate exit
can still cancel unfinished downstream work; forced capture flush does not
guarantee full ASR/MT draining during shutdown.

Further accuracy/review work removed default legacy DSP before recognition,
registered TranslateGemma's actual end-of-turn stop token, and added per-session
audio evidence, exact float-input replay, integrity metadata and a local RTL-aware
web UI. The current recording/default path contracts are in
[Accuracy and review](ACCURACY_AND_REVIEW.md), superseding the baseline's shared
`output.wav`/log paths described above.
