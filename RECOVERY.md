# TTS project recovery — November 6, 2026

Workspace: `~/src/tts-translate`.

**Scope correction:** the user has confirmed substantially later October 2026 work with a web UI, LitLink audio input, Hebrew, and language-direction refinements. Those later conversations and the web implementation have not been recovered. The restored May CLI and the newly reconstructed native bridge are an older baseline. See [later-work knowledge and search limits](docs/LATER_WORK_KNOWLEDGE.md).

## Original Git checkout

Origin: https://github.com/brightidea123/tts-translate.git

The recovery started from `origin/main` at `65be3e4bef2133d5f1be309fe1af42e23c8f6071` (May 9, 2026). All five original commits, three trees, three blobs, and three branch references were recovered through the GitHub connection. Object hashes match GitHub exactly, including the signed merge commit, and `git fsck --full --strict` passes. Direct Git network access was unavailable, so the original object database was restored through the GitHub API rather than a normal network clone. Future fetch/pull commands use the original origin when network access is available.

The recovered original Python blobs and modes are preserved in Git history. The newer Python working file now includes explicitly new stream-ordering and translation-prompt fixes described below:

- `tts-transcribe.py`: newer 7,407-line implementation, blob `b76e7080d6c1cfd668473d75a63e89e47e5dbb0a`.
- `tts-transcribe`: older executable 289-line implementation, blob `29d48f44c6222171cc14fe5e3cc8ceff537b294e`.

## Recovered history and Library sources

The old Codex task is accessible using the current route:
https://chatgpt.com/remote/task_e_697f7ad42714832aa862bb22a08e918e

Its exact title is **Integrate TTS Transcribe version into GitHub**. It imports the initial 289-line script; it does not contain the later rewrite. The original prompt and task details are preserved as described in [the task recovery record](docs/ORIGINAL_CODEX_TASK.md).

ChatGPT Library search recovered eight distinct Python source snapshots, six CPython 3.13 bytecode files, and `apple_vpio_timestamp_bridge.swift`. One duplicate source download is also retained. Exact downloads and checksums are archived outside this checkout at `../tts-recovery-evidence/library/`. The Python snapshots are older than GitHub HEAD; their named improvements are retained or superseded there. Two bytecode files reference source variants that were not recovered; they were not executed or decompiled. See [the comparison and inventory](docs/LIBRARY_RECOVERY.md).

The recovered timestamped Swift capture helper is copied here unchanged. The original active raw-duplex `apple_vpio_bridge.swift` was not found. The file now supplied with that name is a clearly labeled reconstructed derivative implementing the saved Python client's raw PCM/WAV protocol. See [native bridge provenance and limitations](docs/NATIVE_BRIDGE_RECOVERY.md).

## Reconstruction and validation

New files provide setup/design documentation, dependency manifests, native build commands, and the reconstructed raw helper. They are included with the current runtime in the source backup commit. The original GitHub Python implementation remains the recovery baseline.

- All 23 existing Python pipeline checks passed with Python 3.12.14 and NumPy 2.3.5.
- The reconstructed native helper compiled and passed offline tests of raw PCM boundaries/endianness, fragmented WAV messages, real WAV decoding/rate/channel conversion, and malformed input handling.
- The recovered timestamped Swift source passed type checking.

Those initial recovery checks did not start live microphone capture, playback, speech recognition, or translation models. Original model weights, voice assets, local environment versions, audio routing, and any other unpushed work remain unrecovered. The saved Python native playback wrapper's completion/bleed-guard limitation is documented. Recovery does not establish that this is the exact final pre-reset installation.

## New environment restoration and implementation

Subsequent work on November 6 created `.venv` with native CPython 3.12.14 and installed 86 compatible runtime packages, including current MLX 0.32.4, MLX Whisper 0.4.3, MLX LM 0.32.2, MLX VLM 0.7.12 and MLX Audio 0.6.4. Official PyPI wheel hashes were verified; the full resolved hash lock and local `.wheelhouse` support offline reinstallation. These versions were newly researched and resolved, not recovered from the old computer.

New changes correct exact sample retention in the circular buffer, endpoint overlap, final/partial queue races, stale partial publication, and TranslateGemma's structured language-direction prompts. Runtime/model preparation tools and explicit model profiles were added. These changes are new reconstruction work. See [runtime validation](docs/RUNTIME_VALIDATION.md) for the actual checks, model assets and remaining limits.

The managed execution session exposes neither a Metal device nor CoreAudio input devices. MLX initialization fails cleanly with `No Metal device available`; synthetic macOS speech generation produced empty files and was rejected. Live inference, audio routing, latency and multilingual accuracy therefore require testing from a normal local Terminal session.

## November 7 live-runtime repair

After the user reported direct-launch stalls, the active pipeline was refactored
around supervised ASR/MT processes and separate capture/result/playback threads.
Direct launch now prefers the downloaded local models, heavy optional imports
are lazy, keyboard input drains raw UTF-8 correctly, full queues cannot block
typing, worker failures remain visible, and full-duplex playback no longer
resets or mutes capture. These are additional new changes, not recovered code.
See [the implementation and test record](docs/LIVE_RUNTIME_REFACTOR.md).

The subsequent accuracy audit found repeated model end-of-turn markers and a
missing registered stop token, heavy waveform modification before Whisper, and
shared playback/log files across launches. New changes add the corrected stop
condition, streaming tag sanitation, raw-PCM defaults, per-session diagnostics,
full debug/issue-only audio evidence and replay. An alternate local web UI now
uses the same isolated workers with native browser bidi layout; it is new work,
not the recovered October interface. See [accuracy, review and web usage](docs/ACCURACY_AND_REVIEW.md).
