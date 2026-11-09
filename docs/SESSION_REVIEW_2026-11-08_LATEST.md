# Latest saved-session accuracy and latency review

Reviewed six sessions on November 8, 2026, without starting capture, running
model inference, uploading audio, or stopping the user's existing program.
These measurements describe the saved runs before the changes below.

| Session prefix | Direction | Recorded seconds | Nonempty final transcripts |
| --- | --- | ---: | ---: |
| `20261108T074946` | Hebrew → Arabic | 36.52 | 5 |
| `20261108T074634` | Turkish → English | 168.48 | 14 |
| `20261108T060034` | Hebrew → Arabic | 21.02 | 1 |
| `20261108T055631` | Hebrew → Arabic | 202.68 | 16 |
| `20261108T055250` | Turkish → English | 186.68 | 22 |
| `20261108T053520` | Turkish → English | 476.68 | 51 |

## Audio and accuracy evidence

The recordings contain **1,092.06 seconds** of audio. All **829 raw clips and
829 float32 model inputs** passed their saved hash and raw-to-float comparisons.
No clipping, capture-gap requests, writer-rejected capture samples, or hard
speech-boundary cuts were found in these saved sessions. Four requests crossing
a 300-second recording rotation also matched the concatenated adjacent stream
segments exactly. Those four were previously misclassified as unmapped by the
audit script; they were not missing audio.

There were 111 final ASR requests, including 109 nonempty finals. The two empty
finals matched the configured subtitle-credit filter. Source and translation
outputs contained no leaked model control tokens in the reviewed records.

Recognition and translation quality still need improvement. The saved text
contains uncertain Hebrew wording, proper-name cascades from recognition into
translation, and translations that omit or change quantities or numeric
relationships. These are review findings, not a measured error rate: no human
reference transcripts were available, and waveform hashes cannot establish
whether words were recognized correctly. Individual utterance evidence remains
in the local, ignored reports and original recordings.

Two last utterances had nonempty previews but no final ASR result: Hebrew
session `074946`, utterance 6, and Turkish session `053520`, utterance 53. Their
placement at session end and the shutdown code identify premature worker
shutdown as a concrete loss path. Final recognition must process captured tail
audio instead of silently treating a provisional hypothesis as final.

## Measured latency before these changes

Medians for the two newest substantial live runs:

| Measurement | Hebrew → Arabic | Turkish → English |
| --- | ---: | ---: |
| Capture start → first nonempty ASR inference completion | 1.64 s | 1.67 s |
| Partial ASR inference | 0.44 s | 0.46 s |
| Final ASR inference | 0.50 s | 0.55 s |
| Final translation request | 0.85 s | 0.98 s |
| Saved clip end → final translation completion | 1.40 s | 1.61 s |
| Partial translation requests / final translation requests | 0 / 5 | 1 / 14 |

Typical queue waits were 10–25 ms. The longer Hebrew run had final translation
p95 6.23 s and maximum 7.68 s. Six-second ASR previews overlapping translation
took median 0.88 s, versus 0.60 s without that overlap. This is correlation,
not a controlled measurement of GPU contention.

Clip end includes captured endpoint silence. These numbers are not
last-spoken-word-to-browser-paint measurements. MT first-delta and browser paint
timestamps are not persisted. The text log deliberately waits about 1.2 s after
translation completion to allow refinements; its timestamp must not be used as
UI latency.

## Changes justified by the evidence

- **Prevent live translation starvation.** Previously every extension of the
  confirmed prefix restarted a two-second quiet period. Continuous speech could
  therefore postpone all translation until final ASR. Confirmed, unsent words
  now retain their batching deadline while the prefix grows. Actual corrections
  restart stabilization. Minimum word/completeness gates and the two-second
  admission interval remain in force.
- **Avoid wasting optional translation work.** A busy optional lane or pending
  committed work defers another preview instead of repeatedly superseding work
  already computing. Rejected admission does not mark text as sent. Final
  translations and typed replies retain their existing priority.
- **Finish captured tail audio on Stop.** Stop closes capture and finalizes the
  ring's remaining samples before shutting down inference. A bounded grace
  period (default `--shutdown-grace-seconds 6`) lets accepted final ASR and
  translation results finish. Loading,
  unavailable or timed-out workers must be reported rather than silently
  claiming that unfinished audio was finalized. Optional translation previews
  are suppressed while stopping. Each stop records its completed, unavailable
  or timed-out outcome in the session's `shutdown-drain.json`; pending final
  text logs are flushed before exit.
- **Correct saved-session auditing.** The audit reads current session metadata
  from the runtime directory and compares requests across contiguous rotated
  recording segments without hiding gaps or byte mismatches.

These fixes have deterministic regression coverage. Actual latency and quality
after the changes require another real recording; the saved runs cannot measure
an implementation that was not running when they were captured.

Validation completed: **275 Python tests, 19 web tests, 35 embedded pipeline
checks, native protocol self-tests, and whitespace checks passed**. New tests
cover continuous prefix growth, rejection/retry, optional-worker contention,
final priority, sample-exact tail flushing, final ASR → MT delivery before
shutdown, pending callbacks, bounded timeout/error cleanup, final log flushing,
and recording-rotation/configuration auditing.

## Model decisions and comparison with Google

Full MLX Whisper large-v3 and TranslateGemma 12B 4-bit remain selected. The
recorded Whisper effective beam count was always 1. The upstream
[MLX beam-search implementation proposal](https://github.com/ml-explore/mlx-examples/pull/1454)
was still open when checked; raising the requested beam count does not improve
this installed engine's decoding. An alternative final engine needs a measured
comparison before changing defaults.

The hard 256-token translation limit did not explain these runs: CPU-only
tokenization of 109 final translations found a maximum of 141 visible tokens,
with none reaching 240. Increasing that limit or adding prompt text would not
address the demonstrated scheduling bug, and could change throughput or the
model's supported translation-message format.

Google's streaming service returns interim recognition during capture;
[its streaming API documentation](https://docs.cloud.google.com/speech-to-text/docs/v1/speech-to-text-requests)
is a useful behavior target. This local engine repeatedly decodes bounded
Whisper windows rather than using a native streaming recognizer. A model switch
or faster preview setting must be compared on the same audio and references.

To establish comparable or better quality, use the same labeled corpus across
Arabic, Hebrew, German, Turkish, French and Russian, including short answers,
names, amounts, code switching, noise and uninterrupted speech. Measure:

- ASR WER/CER and omitted/repeated words against human references.
- Translation adequacy and preservation of names, numbers, units and negation.
- First visible source and translated text, and end-of-speech final p50/p95
  latency, including endpointing and browser delivery.
- Revision churn, capture loss, and last-phrase completion on Stop.

No Google benchmark, WER/CER score, or best-in-class result has been established.
Google's own
[audio guidance](https://docs.cloud.google.com/speech-to-text/docs/best-practices)
also supports matching the source format and avoiding unvalidated noise
processing. The clean raw-input checks here do not justify introducing an
untested denoiser or relaxing speech gates globally.

Local evidence files: `runtime-reports/recent-performance-audit.json`,
`runtime-reports/recent-accuracy-audit.json`, and
`runtime-reports/translation-token-length-audit.json`.
