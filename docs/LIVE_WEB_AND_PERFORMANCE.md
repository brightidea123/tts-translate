# Live web interactions and performance

Changes made after reviewing session
`20261108T032335.290230Z-pid5822-3606a6d55f` on November 8, 2026.

## Web controls

Select a phrase inside an original or translated message to open its translation
tooltip. The direction comes from that particular text block: selecting the
translated half reverses the direction. This works in incoming conversation and
typed-reply history, including Hebrew/Arabic. Selections must stay inside one
text block; mixed-language-block selections are ignored. Escape or the close
button dismisses the tooltip. The currently selected card is held steady during
live updates, then refreshed when selection is released.

Selections use the running session's translation model. They have their own
result channel, never enter conversation history or TTS, and do not change the
session's language settings. A short debounce avoids translating every dragged
character. Responses are cached, bounded and tied to the current session and
selection ID. Cached selections can be reused; an uncached selection requires a
running session. A busy worker reports that the selection should be retried.

Each pane has an **Auto-scroll** checkbox. Scrolling upward by wheel, touch,
keyboard or scrollbar unchecks it. Appending messages, finishing translations,
or returning to the bottom does not re-enable it. Recheck it to jump to the
latest message and resume following. When history is trimmed, the UI tries to
keep the first surviving visible card at the same screen position.

An unfinished phrase has a darker background in both original and translated
halves. Its normal background returns only when both halves are finalized.
Historical cards remain readable while processing continues below them.

Start with the normal entry point:

```sh
source .venv/bin/activate
python tts-transcribe.py --ui web --debug
```

Restart the Python web launcher after installing these changes, then reload the
browser so both the new API routes and the JavaScript module are loaded.

## What the latest session showed

| Observation | Recorded result |
| --- | --- |
| Input capture | 148.32 seconds; no recorder loss or recorded gaps |
| Saved raw/model inputs | 178 / 172; all hashes, stream positions and raw-to-float conversions matched |
| Accepted final transcripts | 12; all 12 reached the combined translation log |
| Partial ASR requests | 166; about 74.51 seconds total processing |
| Final ASR requests | 12; about 7.21 seconds total processing |
| Partial processing median / p95 | 0.461 / 0.613 seconds |
| Final processing median / p95 | 0.545 / 0.959 seconds |

About 91% of recorded ASR processing was spent on previews. These times exclude
audio-evidence flush, IPC, queue waiting and some other pipeline work. They are
neither pure GPU time nor end-to-end subtitle latency. The combined log is
intentionally debounced by 1.2 seconds, so its timestamp is not the UI timestamp.

One 18.68-second final collapsed to a generic thanks-for-watching phrase while
earlier six-second previews contained substantial narration. Its confidence
proxies looked good. The application now retains a
`final_preview_disagreement` review reason for this pattern, independent of
decoder confidence. It preserves the final and preview evidence rather than
silently combining speculative words into a replacement transcript.

An extreme repetitive preview in that session had compression ratio about 8.52.
Previews that are both repetitive and above the compression limit are now held
for review instead of being sent into live translation. Authoritative finals
and normal spoken emphasis remain preserved.

## Concurrency and module changes

- One process owns each model, with serialized generation. ASR, translation,
  capture, controls, playback and persistence can progress independently.
- Selection work uses a bounded background lane on the existing translation
  worker. Committed live work runs first; live previews run before background
  lookups. Only the newest queued selection is retained. Eviction/rejection is
  reported to its own tooltip. An already-running lookup is not preempted.
- `live` and `m4-max-live` preview cadence is now one second, retaining the
  six-second acoustic window. A busy ASR preview is allowed to finish instead of
  repeatedly superseding its output. Finals retain queue priority. The cadence
  can be overridden with `--streaming-asr-chunk-seconds`.
- `tts_runtime/headless_output.py` owns bounded JSON output: a dedicated writer
  prioritizes command/result packets and coalesces status to at most 10 Hz.
  JSON serialization and pipe writes no longer run while capture/UI code holds
  the dashboard lock. A blocked web reader cannot grow an unlimited snapshot
  queue. Critical-output overflow causes a visible session termination rather
  than silently dropping command replies.
- `selection_requests.py` owns bounded request history, timeouts and LRU caching;
  `selection_inference.py` owns the process-safe job/result types;
  `web/ui-interactions.js` owns selection and scroll behavior.
- New `inference-timing.jsonl` records queue wait and request duration by stage,
  job, live/background lane and source key. It contains no phrase text. These
  timings allow later tuning of latency instead of guessing from log delays.

The one-second preview cadence is a workload/refresh tradeoff, not a measured
speedup. No new GPU benchmark was available in this managed session.

## Validation and the comparison target

`make check` exercises Python/native checks. `make check-web` uses Node for
frontend interaction tests; no Node runtime is needed to run the web UI itself.
The HTTP tests use in-process ASGI, and a real subprocess JSONL test verifies the
selection path/cache without a model. Tests cover queue priority, result/session
isolation, blocked output, stale selections, mixed RTL/LTR text and scroll
pause/resume. The managed browser could not connect to the localhost preview,
so visual browser and real-model tooltip checks remain outstanding.

To establish performance better than Google, use the same audio/reference
transcripts across Arabic, Hebrew, German, Turkish, French and Russian and
measure word/character error, omitted words, translation adequacy, first-preview
and final-display p50/p95 latency, revision churn and capture/drop counts.
Include short answers, code-switching, proper nouns, noisy calls and music as
separate conditions. Compare model outputs and human reference ratings directly;
decoder probabilities do not establish translation quality. No Google
comparison, WER/CER result or superiority claim has been produced yet.
