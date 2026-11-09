# TTS Translate

**Recovery scope:** this checkout restores the older May CLI baseline. The user has confirmed a later October web UI with LitLink audio-source work, Hebrew, and language-direction refinements. Its conversation history and implementation remain missing; [the later-work knowledge record](docs/LATER_WORK_KNOWLEDGE.md) preserves the confirmed context and distinguishes it from recovered evidence.

An interactive terminal translator for incoming call audio and typed replies. By default, it transcribes Turkish speech into Turkish text, translates it into English, and turns typed English replies into spoken Turkish. Source and target languages are configurable.

The current entry point is [`tts-transcribe.py`](tts-transcribe.py). The obsolete extensionless `tts-transcribe` implementation has been removed; its original version remains in Git history.

**Accuracy/review update:** generation now stops on TranslateGemma's end-of-turn
token, model tags are filtered during streaming, Whisper receives raw PCM by
default, and each launch has isolated audio/log paths. `--debug` records the full
input; normal runs save flagged clips. The alternate RTL-aware web interface is
`python tts-transcribe.py --ui web`. See [recording, replay and web UI usage](docs/ACCURACY_AND_REVIEW.md).

**November 8 session review:** corrected short-word loss, suppression of repeated
speech, destructive clause cleanup, wall-clock-dependent endpointing and Whisper
control leaks. Accepted source text now has an independent decision log. See
[the evidence and remaining accuracy limits](docs/SESSION_REVIEW_2026-11-08.md).

**Latest six-session review:** saved audio/model-input integrity checks passed,
but confirmed-prefix translation could wait indefinitely during continuous
speech and Stop could abandon the last phrase. Live translation now batches
growing confirmed text without restarting its deadline, defers optional work
behind committed requests, and retries rejected admission. Stop finalizes
captured tail audio and allows up to `--shutdown-grace-seconds 6` for accepted
final transcription/translation; incomplete drains are explicit in the session's
`shutdown-drain.json`. See [measured timings, fixes and remaining model-quality
limits](docs/SESSION_REVIEW_2026-11-08_LATEST.md).

**Pause timing review:** five newer recordings exposed a silence estimate that
counted the same growing gap repeatedly and a text-stability clock that counted
ASR results instead of elapsed audio time. Both are corrected. Only fresh text
can accelerate a boundary; ellipses retain an unfinished-clause pause, and the
18-second soft limit now respects the ordinary pause and hold. See
[measurements, regression coverage and remaining accuracy questions](docs/PAUSE_REVIEW_2026-11-08.md).

**Hebrew → Arabic follow-up:** two long clips collapsed into generic thank-you
finals despite substantial previews. Conflicting outro previews are now held,
and collapsed finals receive one bounded acoustic recovery pass with timestamped
word ownership. Unresolved phrases preserve provisional words as “Review needed”
instead of committing a suspect replacement. Long-prefix translation refreshes
are paced by growth, and optional MT defers to pending final recognition. See
[session evidence, recovery and concurrency limits](docs/SESSION_REVIEW_2026-11-09_HE_AR.md).

**Full-debug follow-up:** a source-script rejection can now trigger acoustic
language verification before discarding a possible language switch. Verified
phrases retain their source language through MT, UI and logs. Short generic
sign-offs also receive an additional check; unresolved wording stays marked for
review. See [the missing-translation and thank-you investigation](docs/SESSION_REVIEW_2026-11-09_DEBUG.md).

**Repeated-final follow-up:** strong decoder repetition now receives bounded
audio recovery before translation. Unverified loops remain review cases, while
ordinary repeated speech is preserved. Named input preferences can be saved in
`settings.local.json`; capture diagnostics retain the actual resolved device.
See [the repetition and input-route review](docs/SESSION_REVIEW_2026-11-09_REPETITION.md).

**Live web update:** selecting original/translated phrases opens a translation
tooltip; each pane has a persistent Auto-scroll control; unfinished cards use a
darker background. Selection requests share a bounded background model queue,
and headless status output is coalesced off the UI thread. See
[web behavior, latest-session timings and concurrency changes](docs/LIVE_WEB_AND_PERFORMANCE.md).

## Start the responsive live runtime

Direct launch now selects full Whisper large-v3 + TranslateGemma, SoundDevice
and the ordered ring pipeline. Fetch the full Whisper model once after switching
from Turbo (the existing Turbo weights cannot satisfy a large-v3 request):

```sh
source .venv/bin/activate
python scripts/prepare_models.py --download --model whisper
python tts-transcribe.py
```

The default is Turkish incoming speech → English text, with typed English →
spoken Turkish. Select the actual call input and any supported language pair:

```sh
python tts-transcribe.py --list-devices
python tts-transcribe.py --input-device 'LitLink Audio Bridge' --source-language AR --target-language HE
```

The dashboard has persistent **CAPTURE**, **ASR**, and **TRANSLATION** rows.
Capture shows received audio duration, level, speech/silence and dropped samples.
Model rows show loading/ready/busy/failed and backlog. Capture and typing start
without waiting for model warmup. Heavy inference runs in two supervised model
processes; capture, coordination and playback run independently. A failed worker
or full committed mailbox rejects new input visibly; a rejected typed reply
stays editable. Pasted Unicode text and Enter are drained without waiting for
another keystroke. Full-duplex playback does not reset or mute capture.

Live source previews default to `--partial-display-policy latest`: the first
decoded words appear immediately, including the provisional tail. Both the
terminal and web UI show provisional words in red with a solid underline. The
steadier prefix remains in normal text while only the tail changes. Overlapping
windows align across punctuation and casing changes, and temporarily shorter
previews retain earlier words. Words remain visible during the next decode;
final source ASR can correct the complete phrase. Before any words are available, the placeholder is
only `..`. Empty cards expire independently of request completion after capture
and pending final ASR finish (the default grace period is 2.2 seconds from card
creation); stopping a session removes unused empty cards immediately. Actual
source or translated words are preserved. This styling reflects agreement between previews, not a measured
accuracy probability. The live profile requests previews at roughly one-second
audio intervals; display still waits for that decode to finish. Use
`--partial-display-policy stable` for a conservative view (also the default for
the `accurate` profile). Partial translation keeps its separate stable-prefix
policy.

Each web **PHRASE** heading shows its local capture start time and clip duration
in seconds (for example, `14:32:08 · 3.4 s`). Duration grows during the live
phrase and is retained with the finalized text. It measures the captured clip,
including pauses and onset/ending silence, rather than model processing time.

Missing Metal, missing model files, failed synthesis and missing matching voices
produce visible errors. Worker details are local in each
`runtime-reports/sessions/<session ID>/` directory: `pipeline-events.jsonl`,
`asr-worker.log` and `translation-worker.log`. Downloads remain disabled unless explicitly allowed.
`--startup-timeout`, `--inference-timeout`, and `--inference-queue-size` control
worker limits. Finite queues cannot make an overloaded model process every
sample in real time: obsolete previews can be replaced and capture overruns are
reported. The application selects one source per process. See
[the refactor and validation record](docs/LIVE_RUNTIME_REFACTOR.md).

The original Git history is restored at commit `65be3e4bef2133d5f1be309fe1af42e23c8f6071`. Historical Library files and the initial Codex task were also recovered. Setup documentation, dependency manifests, and the raw native audio helper were reconstructed separately. See [`RECOVERY.md`](RECOVERY.md) for provenance, [`docs/ORIGINAL_CODEX_TASK.md`](docs/ORIGINAL_CODEX_TASK.md) for the recovered task, [`docs/LIBRARY_RECOVERY.md`](docs/LIBRARY_RECOVERY.md) for historical source comparisons, and [`docs/RECOVERED_DESIGN.md`](docs/RECOVERED_DESIGN.md) for architecture and backend contracts.

## Set up Python

The restored `.venv` now contains native **Python 3.12.14** and 86 resolved runtime packages. The current MLX stack requires **Python 3.12, macOS 26+ and arm64**. Core diagnostics can run on Python 3.10+ with NumPy; the macOS Command Line Tools Python 3.9 is too old.

Current installed versions are MLX/MLX Metal **0.32.4**, MLX Whisper **0.4.3**, MLX LM **0.32.2**, MLX VLM **0.7.12**, MLX Audio **0.6.4**, Transformers **5.22.1**, PyTorch **2.15.0**, NumPy **2.5.4** and SoundDevice **0.5.6**. Versions were checked and resolved together on November 6, 2026. [The complete lock](requirements-macos-arm64-py312.lock.txt) pins all packages and official wheel SHA256 hashes; `.wheelhouse` holds the verified local cache.

```sh
cd ~/src/tts-translate
source .venv/bin/activate
python -m pip check
make check
python scripts/doctor.py --devices
```

To reinstall from the local cache into a Python 3.12 virtual environment:

```sh
python -m pip install --no-index --find-links .wheelhouse --require-hashes -r requirements-macos-arm64-py312.lock.txt
```

On a connected machine without the cache, omit `--no-index --find-links .wheelhouse`; hashes remain required. `requirements.txt` below installs diagnostics only. [Runtime validation and model preparation](docs/RUNTIME_VALIDATION.md) contain the full run/test procedure.

**Execution limit:** this managed Codex session exposes neither a Metal GPU nor CoreAudio input devices. Package consistency and offline tests pass, but actual MLX inference and live LitLink routing require a normal Terminal session. The doctor reports this failure explicitly rather than claiming model readiness.

These setup commands are reconstructed instructions, not a recovered installer. With Homebrew available:

```sh
brew install python@3.12
cd ~/src/tts-translate
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the existing local checks before installing model backends:

```sh
python tts-transcribe.py --help
python tts-transcribe.py --self-test
python tts-transcribe.py --dump-config
```

Help and configuration output go to **stderr**. To save the effective configuration:

```sh
python tts-transcribe.py --dump-config 2> effective-config.json
```

These commands do not initialize models or open audio devices. New regression checks cover ring sample retention, queue ordering, endpoint overlap, stale partial races and TranslateGemma prompts. These synthetic checks do not establish live microphone, translation, or speech quality.

## Apple Silicon live setup

The starting profile uses full Whisper Large v3 for source-language transcription and TranslateGemma 12B, 4-bit for translation. It explicitly selects SoundDevice capture, the ordered ring engine and energy VAD. The baseline weights total about 9.7 GB; peak memory also includes workspaces/caches. The local model folders are preferred over Hub IDs, and launching is offline by default.

```sh
source .venv/bin/activate
python scripts/prepare_models.py
# Explicitly fetch pinned snapshots only if assets are missing:
python scripts/prepare_models.py --download
python scripts/run_profile.py baseline --source AR --target HE --input-device 'LitLink Audio Bridge'
```

`large-v3` resolves to
[`mlx-community/whisper-large-v3-mlx`](https://huggingface.co/mlx-community/whisper-large-v3-mlx),
whose weights are about 3.08 GB. The preparation tool pins its revision and
checks the published weight SHA-256. It installs into
`models/whisper-large-v3-mlx/`; the runtime also reuses complete local Hub caches
and an existing full-model `models/whisper-large-v3/` directory. Incomplete
project folders do not mask a valid Hub cache. Missing assets produce a
preparation command before warmup, including with `--no-warmup`. To explicitly
use Turbo instead, keep `--whisper-model large-v3-turbo`; its preparation choice
is `--model whisper-turbo`.

The installed packages and model selections are new restoration work, not recovered historical versions. [Model profiles](configs/model-profiles.json) also provide full Whisper Large v3 plus TranslateGemma 27B for quality comparison, and optional Gemma 4 E4B audio ASR. No universal quality or latency ranking has been measured on this machine. The recovered source names Gemma; Gemini cloud models are a different product. TranslateGemma now receives its required structured source/target language messages, while Whisper uses `task=transcribe` and a separate translation stage.

Use `python scripts/run_profile.py baseline --source HE --target FR --dump-config` to inspect the effective setup, or `--dry-run` to print a command. Extra core flags go after `--`, for example `-- --no-session-logs`. Run `python scripts/generate_fixtures.py --generate` and the explicit model smoke tools described in the validation guide from a normal Terminal with Metal/audio access.

Use `--dump-config` with the same flags to inspect this configuration without starting the live session. Use `--list-devices --audio-io sounddevice --vad-backend energy` to inspect available devices after installing SoundDevice; `--input-device` selects an input by name or index. Allow microphone access for the terminal application when macOS requests it. If PortAudio cannot be loaded, check the SoundDevice installation; a system PortAudio installation can be supplied with `brew install portaudio`.

Prefer the exact **input name** for LitLink: numeric device indices can change when audio devices reconnect. The web UI refreshes the device list in a fresh process and starts capture by name. An old numeric selection or missing device requires choosing an input again; it never silently switches to a microphone. Capture checks input channels and 16 kHz mono support before starting model workers, and reports available inputs if that check fails. Restart the web server after updating, refresh its page, then refresh Audio input and select the intended LitLink track by name. `Invalid number of channels` for an output-only index is a device-selection error; rebuilding the native helper does not fix a SoundDevice selection.

The script captures **one selected input per process**. LitLink must route the desired application's audio into that device; selecting its name does not configure tracks or the call application. Simultaneous sources and hardware-clock synchronization are not implemented. Sequenced PCM enters a bounded capture buffer; accepted committed ASR work stays ordered while obsolete partials may be discarded. Capture overruns are reported as gaps rather than silently joining disjoint audio. A full committed mailbox rejects new work visibly rather than blocking capture or typing.

## Use the terminal interface

| Mode | Incoming source-language audio | Typed target-language replies |
| --- | --- | --- |
| `--mode audio` | Transcription and target-language text | Disabled |
| `--mode text` | Disabled | Translation and source-language speech |
| `--mode both` (default) | Transcription and target-language text | Translation and source-language speech |

In `both` mode, Enter sends a typed reply; Enter on an empty input forces an audio flush. Escape cancels a nonempty input, then exits when the input is empty. Exiting cancels unfinished ASR/translation work; to retain the last utterance, force a flush and wait for its final output before exiting. The program needs an interactive TTY for live keyboard mode.

Supported language codes are `AR`, `DE`, `EN`, `ES`, `FR`, `HE`, `RU`, and `TR`. `SP` is an alias for Spanish; additional three-letter aliases are listed in the source. Source and target must differ. Speech backend/model/voice support can vary by language.

Profiles are `live` (default), `balanced`, `stable`, `accurate`, and `m4-max-live`. Explicit command-line flags override profile defaults. **`m4-max-live` selects Apple VPIO audio and requires building the reconstructed bridge or supplying explicit audio overrides.** See the design document for the profile comparison.

Session logs are enabled by default inside `runtime-reports/sessions/<session ID>/`: finalized microphone transcription/translation in `transcription.log`, typed text/translation in `tts.log`, and synthesized playback in `reply.wav`. Use `--no-session-logs` to disable text logs. Explicit output/log path flags override these isolated defaults. Audio evidence is separate under `recordings/<session ID>/`. Model downloads, logs, and generated audio are local runtime data, not recovered project assets.

## Optional backends

The MLX routes are installed and pinned in the current lock. Other optional routes below remain source-derived package requirements and have not been exercised here.

| Capability | Backend flag | Additional requirement |
| --- | --- | --- |
| MLX speech recognition | `--stt-backend mlx-whisper` | `mlx-whisper`, model weights |
| Gemma audio recognition | `--stt-backend mlx-vlm-gemma` | `mlx-vlm`, compatible audio model |
| CPU Whisper | `--stt-backend faster-whisper` | `faster-whisper`, model weights |
| whisper.cpp recognition | `--stt-backend whisper-cli` | External `whisper-cli`, local GGML model |
| External dictation | `--stt-backend dictate-stt` | Compatible external `dictate-stt` command |
| MLX translation | `--translation-backend mlx-lm` | `mlx-lm`, model weights |
| Gemma translation | `--translation-backend mlx-vlm-gemma` | `mlx-vlm`, compatible model |
| NLLB translation | `--translation-backend nllb` | `torch`, `transformers`, NLLB tokenizer/model assets; tokenizer setup may require `sentencepiece` |
| WebRTC VAD | `--vad-backend webrtc` | `webrtcvad-wheels` (imports as `webrtcvad`) |
| Silero VAD | `--vad-backend silero` | `silero-vad`, `torch`; optional `onnxruntime` |
| Energy VAD | `--vad-backend energy` | NumPy only |
| macOS speech | `--tts say` | Installed macOS voice, `say`, `afconvert`, `afplay` |
| Kokoro speech | `--tts kokoro` | `kokoro-onnx`, local model and voice bundle |
| Piper speech | `--tts piper` | Compatible `piper` CLI, local Turkish voice model |
| Edge speech | `--tts edge` | `edge-tts` CLI, network access, `ffmpeg` or macOS `afconvert` |

Partial and final ASR lanes can be configured separately. Optional Apple Speech partials and whisper.cpp/Core ML finals require external helpers/assets; their contracts are documented in [`docs/RECOVERED_DESIGN.md`](docs/RECOVERED_DESIGN.md). All flags and their descriptions are available through `--help`.

## Native audio helper

The original timestamped capture source was subsequently recovered as `apple_vpio_timestamp_bridge.swift`. Its framed `APCM` output does not match the active Python reader, and it has no playback implementation. The missing original raw-duplex helper was not recovered; [`apple_vpio_bridge.swift`](apple_vpio_bridge.swift) is now an explicitly labeled reconstructed derivative. Provenance and its remaining limits are in [`docs/NATIVE_BRIDGE_RECOVERY.md`](docs/NATIVE_BRIDGE_RECOVERY.md).

Build and check the reconstructed raw helper without opening audio devices:

```sh
make build-native
make check-native
```

`make check` prefers `.venv/bin/python` and runs the embedded pipeline checks, runtime-tool tests and native protocol checks. The lock records the newly resolved environment; original pre-reset package pins were not recovered.

The original Python error suggested this direct build command:

```sh
swiftc apple_vpio_bridge.swift -framework AVFoundation -o apple_vpio_bridge
```

The reconstructed source implements that expected binary interface. Offline protocol checks do not establish live capture/playback behavior; the SoundDevice configuration above remains usable without a native binary. The preserved timestamped source must not be built under the raw helper's filename.
