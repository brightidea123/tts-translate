# Repeated sentences in the latest saved session

Reviewed `20261108T213953.109265Z-pid17305-aacb0e91ee`, launched on November 9,
2026 at 00:39 in Istanbul. The review used full-debug recording metadata,
recognition decisions, finalized source/translation logs, inference timings and
the code paths that publish web messages. It did not start capture, run models,
restart the user's process or upload audio.

## Where the repetition originates

The strongest repetitions are already present in **committed source ASR**, then
translated into Arabic. They are not merely repeated painting of a live card.

| Utterance | Final audio | Source pattern | Arabic output pattern | Decoder compression ratio |
| --- | ---: | --- | --- | ---: |
| 7 | 22.72 s | 74 identical interjection tokens plus one fragment | 103 corresponding interjection tokens | 23.47 |
| 13 | 20.18 s | Four-word clause repeated nine times, with a small initial-word variant | Corresponding three-word unit nine times | 5.20 |

Both had `repetitive_transcript` and `high_compression_ratio` flags, but were
accepted as final and submitted to translation. The preview-only repetition
guard did not prevent pathological finals. The long-final consistency guard
also did not handle every decoder loop: a long repetitive result does not look
like the short collapsed thank-you results that initially motivated recovery.

Other repeated text needs more care. Utterance 2 repeats one token four times;
utterance 5 repeats a phrase twice; utterance 20 has a clause appearing twice.
Their translations largely preserve the source repetitions. Without reference
speech these smaller repetitions cannot all be classified as hallucinations.
Deleting repeated clauses globally could remove genuine emphasis, questions or
corrections.

The decoder's average log probability was approximately −0.110 for utterance 7
and −0.530 for utterance 13. Confident-looking decoder metrics therefore did not
exclude a repetition loop. The effective Whisper beam remained 1 despite a
requested final beam of 2 in this installed engine.

## Translation and display findings

The run contains **eight final audio translation jobs and eight unique final
source/translation log entries**, plus only one partial translation job. Thus
frequent retranslation of a growing prefix does not explain these committed
repetitions.

Translation amplified the extreme interjection loop: utterance 7 took 4.980 s
to translate and emitted more repeated tokens than its source. Utterance 13 took
1.644 s and retained the nine repetitions. CPU-only tokenization counted 207
visible translation tokens for utterance 7, below the configured 256-token
limit. The final generation stop reason was not persisted, so the limit should
not be blamed for producing the loop from this count alone.

The live MT protocol sends cumulative text snapshots. The core replaces the
translation field with each snapshot, and the browser assigns its text content;
neither appends a whole cumulative snapshot to the existing phrase. Transport
deduplicates message IDs, and the browser keeps one card per ID. The saved
final logs themselves contain the repetitions, independently of rendering.

This does not prove that every transient preview revision was ideal, but it
identifies an upstream failure that must be addressed before downstream text
deduplication. A repeated final should receive bounded acoustic verification or
recovery rather than being treated as authoritative merely because it is final.

## Capture-route evidence and new diagnostics

This saved session requested `input_device=null` with the SoundDevice backend:
it selected the **system default input**. The old session metadata did not
persist the resolved device name and index. It is consequently not possible to
establish from that setting whether the actual route was a microphone, LitLink
or another virtual input. The current default device also cannot retroactively
establish the earlier route. No route was changed during this review.

The input module now provides a pure `describe_input_device(...)` helper that
describes the dictionary returned by capture preflight. It records the requested
selection, whether default selection was requested, the resolved name/index,
optional default input and host-API metadata, device capabilities and actual
requested stream format. It does not query devices, import SoundDevice or start
capture. Unknown optional fields stay unknown rather than becoming a guess.
The capture startup can persist that description once it has resolved the input.

The session's configured source was Hebrew. A forced decoding-language label
is not independent evidence of the source material's spoken language. Unusual
Hebrew-script wording may also warrant acoustic language verification. This
review did not listen to the speech or establish a different spoken language.

## Corrections from the review

Strong decoder loops now invoke the existing bounded acoustic recovery before
publication or translation, even when the result is long rather than collapsed.
The detector requires a finite compression ratio above 2.9 plus a substantial
contiguous repeated span. It tolerates punctuation and direction marks without
deleting words. A single word needs at least six copies; a multiword cycle needs
at least three copies and nine words. The repeated span must occupy half the
text, or at least 24 words and one quarter. The saved-session policy replay
selects exactly the two catastrophic finals above, retaining smaller repeats.

Recovery uses the same captured samples in bounded timestamped windows. A
window still exhibiting a strong loop invalidates the attempt. Unresolved
output is not committed or translated; earlier usable text remains unverified.
A second check at coordinator publication prevents a legacy or otherwise
unverified loop from bypassing worker recovery. Genuine double repetitions and
short emphasis remain intact; no global sentence-deduplication pass is applied.

The user confirmed that changing from LitLink to Default was unintentional.
This checkout now saves `LitLink Audio Bridge` in ignored `settings.local.json`:

```json
{"input_device": "LitLink Audio Bridge"}
```

Named local preferences survive application restarts. Explicit CLI selections,
including `--input-device=default`, and explicit native Apple capture take
precedence. The web launcher sends an explicit default selection when requested
so a saved preference cannot override it in the child. Numeric indices remain
available for explicit launches but cannot be saved as stable preferences.
Missing or unavailable named inputs fail clearly instead of substituting the
system default.

Capture preflight now persists a description of its resolved input in
`session.json`, effective configuration and web snapshots. Capture status keeps
the resolved device name and index on subsequent updates. Metadata-only commands
do not query hardware. The description comes from the same preflight used to
open capture rather than guessing the device from its requested selector.

## Shutdown and review limits

Graceful Stop completed in 1.252 s with no pending final ASR, unfinished text,
queue work or model callbacks. The demonstrated repetition reached accepted
final text before shutdown; it is not evidence of an abandoned Stop drain.

Audio and decoder diagnostics identify review candidates, not a word-error
rate or human-verified transcription. Actual recognition recovery and any
changed repetition policy need validation on the same saved audio and human
references. Input-description tests use supplied metadata only and never open
audio hardware.

The full debug recording contains 118.14 seconds. All 92 saved raw clips and
83 saved float32 model inputs passed saved hashes, raw-to-float comparisons and
stream mapping, with zero unmapped requests or audit errors. Some raw requests
were rejected before model preparation, so fewer float inputs do not indicate
missing model buffers. Capture reported no rejected samples. The distinct
one-second PCM blocks in the two loop clips provide no evidence of literal
repeated-buffer insertion.

Validation passed: **396 Python tests, 20 web tests, 35 embedded pipeline checks
and native protocol self-tests**. New coverage includes strong-loop recovery,
coordinator quarantine, genuine repetitions, exact input hashes, capture-device
metadata and saved-input precedence. Actual acoustic recovery quality still
needs a post-change recording or model replay. Restart the application to load
the saved LitLink preference and updated web launcher. No existing capture
process was stopped or restarted during this review.
