# Accuracy, recording and live web UI

These changes address the reported token leakage and make future recognition
failures reproducible. They do not establish recognition accuracy without
reviewing the actual captured speech and a reference transcript.

## Confirmed findings and changes

The old logs contain hundreds of repeated `<end_of_turn>` markers per result.
The local TranslateGemma tokenizer stopped on `<eos>` (ID 1), while the model's
end-of-turn token is ID 106. That token is now resolved dynamically, validated
against the tokenizer, and registered as a stop token. Streaming/final text also
passes a known-control-token filter before display, logging or speech. Partial
control-tag prefixes are withheld, and the final detokenizer flush preserves
the last word and its whitespace. Legitimate Unicode, ordinary markup and math
are not broadly stripped.

Whisper was receiving a 20 ms FFT-filtered, pre-emphasized, gain-normalized
waveform. A valid test impulse produced model-input peaks above 7 after that
processing. This is a plausible source of distortion, not proof of the user's
particular recognition error. The new default is the exact captured PCM mapped
to float32 by division by 32768. Whisper applies its own feature extraction.
`--asr-preprocessing legacy` remains available for replay comparisons. Final
ASR is no longer augmented with speculative partial words by default; empty or
stuck partials are closed instead of silently promoted into final text.

No cross-process PCM collision was demonstrated. Separate sessions did share
`output.wav` and log paths, however, allowing one session's playback file to be
overwritten by another. Each launch now gets a unique session ID and default
reply/log/recording paths. ASR requests carry that session ID, a request ID,
source timestamps, sample count and PCM SHA256. The worker verifies the hash
before decoding; the parent rejects wrong-session, stale-request or wrong-hash
results. Captured frames and submitted requests own immutable copies. Capture
gaps create utterance boundaries, and source-span/sample-duration discrepancies
are flagged in provenance rather than silently hidden.

## Record a session

Terminal interface:

```sh
source .venv/bin/activate
python tts-transcribe.py --debug --input-device 'LitLink Audio Bridge'
```

`--debug` defaults to **full input recording**, before VAD/endpointing, model
processing or capture-queue eviction. It also retains ASR raw clips and exact
float32 model inputs while the clip budget permits. Without `--debug`, the
default is **issue clips only**: empty/filtered output, clipping/quiet audio,
decoder probability/compression warnings, unexpected language, source gaps,
rejected final requests or worker errors. These are confidence/issue proxies,
not measured accuracy or WER. Confidently wrong text may not trigger them.

Press **Ctrl-R** in the terminal, or **Save review clip** in the web UI, to mark
a mistake manually. This saves the newest contiguous interval, up to 15 seconds;
it never combines disjoint intervals merely to fill the duration.

Controls:

| Flag | Behavior |
| --- | --- |
| `--audio-review auto` | Full stream with debug, issue clips otherwise; default |
| `--audio-review full` | Explicit full stream recording |
| `--audio-review issues` | Explicit issue clips only, including with debug |
| `--audio-review off` | Disable audio evidence |
| `--recordings-dir PATH` | Recording base directory; default `recordings/` |
| `--issue-audio-budget-mb 256` | Per-writer raw/blob byte budget; existing files are never deleted |

Full audio is 16 kHz mono Int16 WAV, rotated every five minutes. Hardware
overruns, clock resets and recording-queue loss are explicit events and split
the recording; unknown missing input is not invented as silence. The background
writer never performs filesystem I/O in the audio callback. Its queue is
bounded, so a stalled disk can still lose recording packets; rejection counts,
gap intervals and writer errors are preserved in the summary. Full stream
rotation does not impose a total debug-session disk budget.

## Locate and replay evidence

Each launch uses:

```text
recordings/<UTC timestamp>-pid<PID>-<unique ID>/
  session.json
  capture-...-events.jsonl / -summary.json
  capture-...stream....wav / .json
  asr-...raw....wav / .wav.json
  asr-...model-input....f32le / .f32le.json

runtime-reports/sessions/<same session ID>/
  session.json
  asr-worker.log / translation-worker.log / pipeline-events.jsonl
  asr-decisions.jsonl
  reply.wav / transcription.log / tts.log
```

Artifact names include writer/PID/unique identifiers. Capture and ASR writers
can share the session folder without sharing a file handle or filename. Clip
sidecars contain request/source provenance, raw and sanitized text, decoder
metrics when provided, preprocessing, issue reasons, byte hashes and sample
counts. The `.f32le` file preserves the model input exactly, including values
outside [-1,1] in a legacy comparison; it is not quantized back to PCM.

`asr-decisions.jsonl` preserves accepted final source text before translation,
plus filtering/duplicate/translation-rejection decisions. It is disabled with
`--no-session-logs`. Run `python scripts/audit_sessions.py --latest 3` for an
offline integrity and outcome audit. The default skips empty recordings;
`--include-empty` includes failed startups. See the
[November 8 session findings and fixes](SESSION_REVIEW_2026-11-08.md) for confirmed
short-word loss, duplicate suppression and endpoint defects found in the debug runs.

Inspect a saved clip without loading a model or accessing the network:

```sh
python scripts/replay_audio.py 'recordings/<session>/<clip>.wav'
```

Explicitly replay it, or compare the legacy preprocessing:

```sh
python scripts/replay_audio.py 'recordings/<session>/<clip>.wav' --run --source HE
python scripts/replay_audio.py 'recordings/<session>/<clip>.wav' --run --source HE --preprocessing legacy
python scripts/replay_audio.py --prepared-f32 'recordings/<session>/<model-input>.f32le' --run --source HE
```

Sidecars supply the source language when available; `--source` overrides it.
`--asr-model` selects another local model. Replay is offline unless
`--allow-download` is supplied, does not open a microphone/speaker, and disables
new review recording. Decoder-gated or failed input does not count as completed
inference. Earlier debug runs did not save their source audio, so their text
logs cannot reconstruct the missing waveform.

## Alternate web UI

```sh
source .venv/bin/activate
python tts-transcribe.py --ui web
```

Open the printed URL (normally `http://127.0.0.1:8765`), choose source/target
languages and input, then press **Start session**. Use `--ui web --debug` to
preselect full recording, or change the debug checkbox before starting. The UI
respects an explicit `--audio-review` policy and shows that policy beside the
debug checkbox. The server is local-only and uses the same model pipeline.
Each Start launches an
isolated backend process; Stop closes only that owned session. Configuration
changes take effect on the next Start.

The UI streams bounded state snapshots over WebSocket. Incoming speech and
typed replies have separate source/translation blocks with independent text
direction, language fonts and bidi isolation. Arabic/Hebrew are preserved in
logical Unicode order, rather than reversing strings or setting the entire
page RTL. Enter sends, Shift-Enter adds a newline, and IME composition is
respected. Drafts survive rejection/disconnection. Stage status, input level,
recording policy, saved directory, force-flush and manual review controls are
available without opening the terminal dashboard.

Phrase selection now opens a translation tooltip in the opposite direction
for that text block. Auto-scroll can be disabled independently per pane and
turns off when you scroll upward. Active phrases remain darker until finalized.
See [interaction details and performance measurements](LIVE_WEB_AND_PERFORMANCE.md).

## Validation and remaining limits

Tests cover stop registration with the actual local tokenizer, streaming tag
fragments, final-flush whitespace, exact raw/model-input artifacts, recording
rotation/loss/discontinuity, per-session path isolation, request/result integrity,
stale-result quarantine, replay, JSONL control, REST validation, WebSocket
versioning, Unicode/RTL snapshots, and existing concurrency invariants. A real
headless core subprocess accepted Unicode commands and stopped cleanly.

This managed session cannot access Metal/CoreAudio or bind a localhost server.
HTTP/WebSocket checks therefore use in-process ASGI tests. The browser policy
also blocks local-file previews, so visual browser verification was not
completed. Run the web command from normal Terminal for the live UI/hardware
check. No claim of improved measured recognition accuracy is made yet; the
recordings now provide the evidence needed to assess it.
