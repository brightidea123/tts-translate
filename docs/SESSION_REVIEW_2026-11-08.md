# Review of the November 7 debug sessions

Reviewed on November 8, 2026. The recordings cover the sessions started around
22:51–22:55 Istanbul time on November 7 (19:51–19:55 UTC):

| Session ID prefix | Selected direction | Captured audio |
| --- | --- | --- |
| `20261107T195100` / PID 4034 | Turkish → English | 9.58 s |
| `20261107T195113` / PID 4045 | Turkish → English | 217.36 s |
| `20261107T195517` / PID 4074 | Hebrew → Arabic | 180.18 s |

The full large-v3 model loaded in these sessions. Its MLX backend reported
unsupported beam search and used greedy decoding; selecting beam 2 did not
provide a beam-search quality improvement.

## What the saved evidence establishes

All **445 raw ASR clips** hash-match their submitted requests and the exact
corresponding bytes in the full input recordings. All **425 float32 inputs**
match their recorded hashes and equal raw PCM divided by 32768. No recorder
loss, clipping, capture discontinuities or source-span mismatches were found in
these artifacts. This rules out a sample mismatch for the saved requests; it
does not establish that every spoken word was submitted or recognized.

Confirmed application defects:

| Finding | Evidence / effect | Change |
| --- | --- | --- |
| Valid short words discarded | Hebrew final 22 decoded `שלום.` with average log probability about −0.322 and no-speech probability about 0.014; it became empty. | Removed the hidden rejection of all one/two-word strings up to six characters. Configured minimum text thresholds still apply. |
| Genuine repeats suppressed | Hebrew finals 23 and 24 both decoded `מה זה אומר?`, from disjoint audio about 7.88 s apart; only the first reached the combined log. Turkish finals 2/3 also repeated text from separate audio. | Cross-utterance suppression now needs matching text and proven substantial overlap in the same source recording. |
| Legitimate clauses rewritten | The fuzzy cleanup can turn “We need five blue chairs and we need six red chairs” into only the second clause. It also removed doubled emphasis. | Authoritative ASR no longer rewrites repeated/similar clauses. Repetition remains a review signal. |
| Brief endpoint input silently rejected | Trailing endpoint silence reduced the speech-ratio score. Synthetic 100–180 ms words meeting the five-frame minimum were discarded. Ten Hebrew IDs had partials but no saved final. | Speech-ratio gating excludes trailing silence; rejected endpoints retain an explicit reason and review clip. The missing historical finals cannot all be assigned a cause retrospectively. |
| Endpoint depends on processing speed | The 140 ms hold used wall time, so queued audio processed rapidly could merge phrases that realtime delivery split. | Hold advances with captured samples. |
| Abrupt long-speech cuts | Hebrew finals 1/2/3 lasted 18.00/18.22/18.00 s, without spoken overlap; the first two ended in active audio. | After 18 s, prefer a quiet boundary. Keep a 28 s hard bound, explicitly flagging and saving forced boundaries. |
| Whisper controls leaked | Hebrew final 21 and its final log contained `<\|lt\|>`. | Filter Whisper language, task and timestamp controls and retain a review flag. |

The Turkish run produced many `Altyazı M. K.` caption hallucinations that were
filtered (150 saved requests were empty after filtering). Its decoded content
appears song-like, so it is not a clean conversational-speech benchmark.
Filtering those captions does not demonstrate that the remaining lyrics are
accurate. No reference transcript or new GPU inference was available for this
review; WER/CER and actual ending-word accuracy remain unmeasured.

The explicit caption-phrase filter remains enabled by the default anti-repeat
policy. It cannot distinguish a hallucinated caption from someone actually
saying the same phrase. Use `--known-hallucination-filter off` for a comparison
that must preserve those phrases; raw decoder text remains in debug sidecars.

## Additional live changes

Growing-window ASR hypotheses now replace earlier readings of that same source
start. A correction such as “five chairs” → “six chairs” no longer appends both
versions or stays frozen behind an old display lock. Shifted windows still use
the existing provisional merge; the final full-window decode remains authoritative.

The `live` and `m4-max-live` preview context grows to six seconds instead of
1.8/2.2 seconds. This supplies more acoustic context without waiting six seconds
to start: the first preview remains eligible on the existing cadence. It is a
tuning change, not a measured accuracy/latency win. Override it with
`--partial-asr-window-seconds` when comparing recordings.

Accepted source text is written to the new per-session `asr-decisions.jsonl`
**before translation is queued**. Empty finals, proven duplicate-audio results
and rejected translation requests have separate decisions. A translation
failure therefore cannot erase the only recognition record. `--no-session-logs`
also disables this text journal. Decoder diagnostics now record elapsed ASR
processing time and effective beam count for future comparisons.

The hard 28 s fallback can still intersect a word in uninterrupted speech. It
is bounded and reviewable; this change does not claim word-aligned segmentation
for every possible source. Capture and recording remain independent of model
execution.

## Run and inspect

Use the normal command or the web interface with debug audio recording:

```sh
source .venv/bin/activate
python tts-transcribe.py --ui web --debug
```

Review saved artifacts without loading a model, opening a device or using the
network:

```sh
python scripts/audit_sessions.py --latest 3 --output runtime-reports/latest-audit.json
python scripts/audit_sessions.py --session 20261107T195517.199223Z-pid4074-c65880fcf6
```

Latest selection skips zero-audio sessions; `--include-empty` includes failed
startups/smoke sessions. Explicit session selection always includes the named
directory. Console output contains diagnostics and text hashes rather than
copies of the conversation. A partial-only ID is a review flag, not proof that
speech was lost: issue-only recording may omit successful requests.

The original audit reports remain under `runtime-reports/recent-session-evidence.json`
and `runtime-reports/session-audit-debugged.json`. No recordings or historical
transcription logs were rewritten. Tests cover brief words, correction/negation
preservation, proven audio overlap, clock-independent endpointing, quiet versus
hard duration boundaries, control filtering, independent source logs and audit
integrity. GPU replay and human/reference evaluation remain required to measure
recognition quality.
