# Pause and phrase-boundary review

Reviewed five newer saved sessions on November 8, 2026. The analysis used saved
PCM, request provenance, model timings, decision logs and shutdown reports.
It did not start capture, stop the user's process, upload audio or run models.
These recordings predate the endpoint changes described below.

## What the recordings show

| Session prefix (UTC) | Direction | Recorded seconds | Final ASR requests | Ordinary endpoint quiet, median / p90 |
| --- | --- | ---: | ---: | ---: |
| `20261108T152434` | Arabic → English | 434.96 | 37 | 420 / 560 ms |
| `20261108T151817` | Hebrew → Arabic | 282.32 | 40 | 500 / 960 ms |
| `20261108T151345` | Turkish → Arabic | 82.82 | 12 | 680 / 960 ms |
| `20261108T151221` | Turkish → Arabic | 71.82 | 14 | 560 / 960 ms |
| `20261108T135826` | Turkish → Arabic | 111.46 | 15 | 460 / 560 ms |

The five sessions contain 983.38 seconds (16.4 minutes) of audio. The quiet
measurements replay the configured energy gate over 20 ms PCM frames. They
measure the trailing quiet included in ordinary final clips, including the
hold; duration-limit cuts and Stop flushes are excluded from those percentiles.
Energy-gate quiet is not a human-labeled word, sentence or paragraph boundary.

In the Arabic run, interior quiet runs of at least 120 ms had median 170 ms and
p90 400 ms. For Hebrew these were 300 ms and 698 ms. Speech resumed within
750 ms of the last voiced frame at 24 of 29 measurable ordinary Arabic
boundaries and 14 of 39 Hebrew boundaries. These overlaps between internal
pauses and clip endings make a universal shorter silence limit inappropriate.
The Arabic run also made seven cuts at the old 18-second soft limit.

The latest four sessions received a separate byte-level audit: all **664 raw
request clips and 664 float32 model inputs** matched their saved hashes and
raw-to-float conversions. All raw clips mapped exactly to the saved stream;
five requests spanned a recording-file rotation. No capture samples were
reported rejected. The Turkish `151221` run nevertheless had two clipped-audio
request flags: matching bytes do not establish clean or intelligible speech.

Every reviewed session reports a completed Stop drain with no remaining final
ASR, unfinished text, queue work or model callbacks. The latest four drained
in 0.252–1.776 seconds. The newer Arabic session produced 45 partial translation
requests covering 26 utterances, confirming that live translation was no longer
entirely postponed until final recognition in that run.

## Corrected timing defects

1. **A growing silence trained its own deadline.** The old estimator sampled the
   same pause again every 20 ms as 20, 40, 60 ms and so on. Logs show a single
   pause moving its threshold from 660 to 798 to 820 ms. It now observes a pause
   once, when speech resumes, and retains the learned estimate across phrases.
   Short closures cannot train it below the configured baseline. An unfinished
   trailing silence does not move its own acoustic deadline. The legacy path
   also resets its gap counter when speech resumes, so separate pauses do not
   accumulate.
2. **Text-stability milliseconds were result counts.** An unchanged ASR result
   previously added only 20–40 ms to a value compared with 220/420/900 ms gates,
   despite results typically covering another second of captured audio. The
   clock now measures elapsed captured audio since the last text change, with
   monotonic elapsed time as a fallback for results lacking source timestamps.
   Regressive audio coverage cannot roll the clock or words backward.
3. **Older text could end newer speech.** Text-based early finalization now
   requires a validated result covering the latest voiced frame, allowing only
   the existing two-sample timestamp tolerance. Text, stability and coverage are
   read together under one lock. Older hypotheses leave the acoustic deadline
   in control. They cannot use a newer result's timestamp by interleaving two
   separate state reads.
4. **Ellipses looked like full stops.** `...` and `…` now indicate unfinished
   text. Arabic question punctuation and continuation words for Arabic, Hebrew,
   German, French and Russian supplement the existing English/Turkish hints.
   Repeated agreement alone cannot turn an unfinished clause into a reason to
   shorten the acoustic pause. These are conservative text hints, not a
   grammatical parser or an ASR-confidence measurement.
5. **The soft duration limit bypassed normal endpoint protection.** Reaching
   18 seconds previously allowed a final cut on a 240 ms gap without the normal
   hold. It now waits for the ordinary adaptive pause and hold, with a 240 ms
   minimum at that soft limit. The 28-second safety cap remains. If a newer
   hypothesis raises the pause threshold during a pending final, the hold
   restarts when that later threshold is reached. Returning speech cancels the
   pending final and remains in the same clip.
6. **A faster endpoint could reject a valid short answer.** Quiet tail samples
   are retained until the configured minimum decoder input span is reached.
   A 100 ms valid answer is not discarded merely because its earlier pause
   deadline made the whole input fall just below that minimum.

Configured silence limits, the 140 ms live-profile hold, model choices, capture
ordering and inference queues are unchanged. Actual endpoint waits can change
because the broken estimator and stability clock no longer distort them.

Final request provenance now includes `trailing_silence_ms`,
`endpoint_threshold_ms`, `finalization_hold_ms`, `completed_pause_count`,
`completed_pause_estimate_ms` and `endpoint_text_fresh`, making the next review
able to distinguish acoustic timing from text-based decisions directly.

## Remaining quality and performance questions

The Arabic run contains two finals substantially shorter than their earlier
previews (utterances 3 and 5). One triggered `final_preview_disagreement`.
This warrants replay and a reference transcript; pause timing or buffer
corruption alone does not explain it. Earlier hypotheses may also be wrong,
so they are not promoted into authoritative final text automatically.

Saved preview coverage was often about a second behind captured audio when a
final boundary was chosen (median 1.16 s Arabic, 0.98 s Hebrew). Availability
was estimated using saved artifact timestamps, which may include writer delay.
This supports checking freshness rather than trusting punctuation from the
latest visible sentence. Median final ASR compute was 0.837 s Arabic and
0.484 s Hebrew; median Arabic final translation compute was 1.166 s. These are
model request timings, not last-word-to-browser latency.

All five runs used energy VAD; the optional neural VAD package was unavailable.
Energy alone cannot reliably distinguish quiet syllables from background noise
or establish paragraph boundaries. A neural-VAD comparison should use the same
saved audio and human annotations before changing defaults.

Validation passed: **289 Python tests, 19 web tests, 35 embedded pipeline
checks and native protocol self-tests**. Regression cases cover completed-pause
learning, source-clock stability, old-result rejection, fresh semantic hints,
ellipsis/RTL punctuation, short answers, resumed speech, changing deadlines,
soft-limit word gaps and exact preservation of captured samples. No new live
latency benchmark or WER/CER measurement was performed; quality gains still
require a post-change recording and reference transcripts.

Local evidence remains ignored by Git: `runtime-reports/pause-boundary-audit.json`,
`runtime-reports/new-pause-session-integrity.json` and
`runtime-reports/new-session-timing-audit.json`. No existing process was
restarted; a new session loads the updated endpoint code.
