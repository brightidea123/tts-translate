# Runtime validation, 2026-11-06

The restored environment uses CPython 3.12.14 on an Apple M4 Max, arm64,
Darwin 27.2.0. Its 86 resolved runtime packages were installed successfully,
and pip was upgraded to 26.3. This is a newly resolved environment; the
original pre-reset package versions were not recovered.

[`requirements-mlx.txt`](../requirements-mlx.txt) records the exact backend
roots, including the optional MLX VLM/audio packages. The complete
[`requirements-macos-arm64-py312.lock.txt`](../requirements-macos-arm64-py312.lock.txt)
pins all 86 runtime packages plus pip. It targets Python 3.12, macOS arm64,
and macOS 26 or newer because of the selected MLX native wheels.

The lock was generated from `.wheelhouse/install-report.json`. Every selected
wheel hash was checked against both its actual contents and the official PyPI
SHA256 recorded in `.wheelhouse/manifest.json`. The `mlx-audio[stt]` extra is
retained. The resolved constraints select fsspec 2026.7.0 and mpmath 1.3.0;
newer downloaded candidates were incompatible with other dependencies.

Reproduce the installation from the project directory with the verified local
wheelhouse available:

```sh
.venv/bin/python -m pip install --no-index --find-links .wheelhouse --require-hashes -r requirements-macos-arm64-py312.lock.txt
.venv/bin/python -m pip check
```

The wheelhouse is a local cache. Keep it alongside the lock for fully offline
reinstallation. Model weights, tokenizer assets, repository revisions and
model checksums are separate from Python dependencies.

## Checks completed

| Check | Observed result |
| --- | --- |
| `.venv/bin/python -m pip check` | No broken requirements found |
| Offline lock installation with `--dry-run --ignore-installed --require-hashes` | Passed; all 87 locked packages resolved and verified without changing installed packages |
| `.venv/bin/python tts-transcribe.py --self-test` | All 35 fast pipeline checks passed, including ring wrapping, committed audio ordering, SoundDevice callback sequencing/ADC clocks, capture gaps, forced-tail ownership, endpoint boundaries and translation prompt direction |
| `make check-native` | Reconstructed helper passed raw PCM framing/endianness, fragmented WAV messages, real WAV decoding/conversion, and invalid/truncated input checks |
| `.venv/bin/python -m unittest discover -s tests -v` | All 37 tests passed, including child-process failure, native-abort reporting, timeout handling, model shard completeness, model preparation, and Unicode prompt handling |
| `.venv/bin/python scripts/doctor.py --json` | Complete diagnostic JSON; exit status 1; model-smoke readiness false |

The Python pipeline checks use synthetic inputs and model stubs. The native
helper checks do not open a microphone or speaker. These results establish
dependency consistency and the tested processing/protocol behavior.

## Metal restriction in this execution context

The doctor successfully imports NumPy and SoundDevice. MLX Whisper, MLX LM
and a direct `mlx.core` probe each fail with:

```text
ImportError: [metal::load_device] No Metal device available.
```

The MLX error identifies headless, sandboxed or virtualized macOS sessions as
possible causes. The selected MLX and MLX Metal versions match, and the native
extension, library and Metal assets are installed. The observed limitation is
GPU availability in the current execution context; package installation alone
does not establish that MLX can execute here.

Each doctor import, Metal calculation and optional device query now runs in
its own subprocess with a 30-second timeout. A failed native initialization
can otherwise leave nanobind state behind, causing a later import in the same
process to abort with a duplicate-type error. Isolation preserves the original
failure and records process errors without terminating the diagnostic. Skipped
native checks cannot produce a positive model-smoke readiness result.

No speech recognition or translation model inference, language accuracy,
throughput, live microphone capture, playback, LitLink routing or long-running
latency has passed here. Run the doctor and explicit model smoke tests in a
session with accessible Metal before evaluating live behavior. SoundDevice
enumeration returns zero devices and default indices `[-1, -1]` in this managed
session. macOS `say`/`afconvert` returned success but produced empty Arabic and
Hebrew speech files; fixture validation rejected these files. No microphone
capture or playback occurred.

## Downloaded baseline model assets

Both baseline models are stored under the ignored `models/` directory:

| Model | Repository revision | Weight files |
| --- | --- | --- |
| `mlx-community/whisper-large-v3-turbo` | `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | `weights.safetensors`, about 1.61 GB |
| `mlx-community/translategemma-12b-it-4bit` | `f3dcfd54df14672fbcf0731086fb47a797a943ae` | Two shards, about 6.62 GB total |

Downloads used official Hugging Face links through the browser because shell
networking could not resolve remote hosts. All three weight SHA256 values match
the repository file pages. Exact configuration/tokenizer text was saved from
the raw pages and checked after transfer. The local manifest is
`models/model-manifest.json`; it records every retained file's byte count/hash.
The folders contain the inference assets, rather than full repository clones.
TranslateGemma uses its SentencePiece `tokenizer.model`; the optional large
`tokenizer.json` was not downloaded. Actual local `AutoTokenizer` loading worked.

All model safetensors headers are readable without GPU access: Whisper has
587 tensors; the translation shards have 1,032 and 268. The translation index
matches the exact key set and shard assignment. The real TranslateGemma
tokenizer rendered and tokenized all 30 directed AR/HE/DE/TR/FR/RU prompts,
preserving the original Unicode input. This validates tokenizer/template
integration, not generated translation quality. Reports are in ignored
`runtime-reports/model-file-integrity.json` and `tokenizer-30-pairs.json`.
`managed-inference-probe.json` records a real two-stage invocation with a valid
synthetic silence WAV: ASR initialization fails for unavailable Metal,
translation is skipped to avoid retrying a failed native initialization, and
`model_inference_run` is false. The silence file is explicitly labeled as a
runtime probe rather than a speech/recognition fixture.

## Continue from normal Terminal

```sh
cd ~/src/tts-translate
source .venv/bin/activate
python scripts/doctor.py --devices
python scripts/generate_fixtures.py --generate
python scripts/smoke_models.py --asr-model models/whisper-large-v3-turbo --audio fixtures/generated/he.wav --source HE --require-nonempty-asr --output runtime-reports/asr-he.json
python scripts/smoke_models.py --translation-model models/translategemma-12b-it-4bit --all-pairs --output runtime-reports/translation-30-pairs.json
```

Repeat the ASR test with each generated `ar`, `he`, `de`, `tr`, `fr`, `ru` WAV
and matching `--source`. These synthetic voices test the path; representative
real speech is needed for accuracy and latency assessment. Outputs are marked
as smoke results and require human quality review. The smoke tools default to
offline access; no API key or cloud inference is required.

After the doctor reports Metal and the desired input device:

```sh
python tts-transcribe.py --list-devices --audio-io sounddevice --vad-backend energy
python scripts/run_profile.py baseline --source AR --target HE --input-device 'LitLink Audio Bridge'
```

Choose the exact device name or index from the list. Configure LitLink's source
application/tracks separately. The launcher uses SoundDevice, ordered ring
capture and energy VAD explicitly, with local models preferred. It can print
the command with `--dry-run` or inspect core configuration with `--dump-config`.
Additional core flags go after `--`, such as `-- --no-session-logs`.

The ring pipeline processes one input per process. It does not merge several
independent source clocks. Bounded capture protects memory; capture drops or
device overruns require visible gaps and utterance boundaries. Final work is
preserved in submission order, so sustained inference slower than incoming
speech can still build a committed backlog. No measured end-to-end latency or
universal "best model" claim follows from selecting these profiles.

Escape retains the CLI's immediate-exit behavior: unfinished ASR/MT work can
be canceled during shutdown. The new capture close/flush does not promise to
drain the entire downstream pipeline. Force a flush and wait for final output
before exiting when the last utterance must be retained.
