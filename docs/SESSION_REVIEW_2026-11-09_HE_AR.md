# Hebrew → Arabic regression review

Reviewed the session launched at `20261108T210056` UTC (November 9, 2026,
00:00 in Istanbul), comparing it with the earlier Hebrew session
`20261108T151817`. This review used saved request diagnostics, decision logs,
worker timings and flagged audio. It did not start capture, run models,
restart the user's process or upload audio.

## Confirmed recognition regressions

Two final requests replaced much longer previews with the same two-word
thank-you message:

| Utterance | Final input duration | Earlier retained preview | Final output |
| --- | ---: | ---: | ---: |
| 6 | 24.64 s | 19 words | 2 words |
| 8 | 20.08 s | 14 words | 2 words |

Both already carried `final_preview_disagreement` flags, but a flag only saved
evidence: it did not prevent the final from replacing the visible source or
being translated. The first decoder result had average log probability
−0.571 and no-speech probability 0.147; the second had −1.090 and 0.042.
Thus a low no-speech threshold alone would not catch these collapses.
These probabilities are decoder proxies, not measured accuracy.

The previous built-in credit filter contained Turkish subtitle, subscribe and
watching-credit phrases. It did not cover their Hebrew counterparts. In
addition, a generic thank-you phrase can be legitimate speech: deleting every
instance would lose valid short answers. The demonstrated failure requires
using audio coverage and disagreement with earlier recognition, followed by
recovery or an explicit uncertain result, rather than a blanket ban on thanks.

Three repetitive high-compression previews were withheld from display for
final decoding. The saved logs therefore establish both preview hallucination
pressure and two harmful final replacements; they do not establish that every
credit-like phrase in the user's audio was hallucinated.

## Longer phrase windows and translation load

| Saved measurement | Earlier Hebrew session | New Hebrew session |
| --- | ---: | ---: |
| Final ASR requests | 40 | 10 |
| Final clip duration, median | 4.47 s | 18.18 s |
| Final clip duration, maximum | 10.44 s | 28.00 s |
| Hard 28-second cuts | 0 | 3 |
| Soft duration cuts | 0 | 2 |
| Partial ASR compute, median / p95 | 0.451 / 0.698 s | 0.637 / 1.352 s |
| Final ASR compute, median / p95 | 0.484 / 0.815 s | 0.939 / 2.688 s |
| Partial audio MT requests | 8 | 21 |
| Partial MT compute, median / p95 | 0.901 / 1.149 s | 1.495 / 3.096 s |
| Final audio MT compute, median / p95 | 0.703 / 1.521 s | 1.584 / 3.990 s |

The new run spent 34.03 seconds computing 21 partial translations, then
20.30 seconds computing 10 final translations. Utterances 1 and 5 each had
five partial translations, consuming about 10.16 seconds each. Their starts
were typically 2.9–4.9 seconds apart. The selected stable prefix is sent in
full each time, retaining translation context but repeating earlier work as
that prefix grows.

The more conservative fresh-text endpoint rule introduced in the previous
review helps explain why clips can grow longer. Saved endpoints here used an
acoustic threshold near 500 ms plus the 140 ms hold, with
`endpoint_text_fresh=false`; stale punctuation could no longer shorten that
pause. The completed-pause estimate remained close to its 360 ms baseline.
This is consistent with longer windows, but the sessions contain different
speech. The table is not a controlled before/after benchmark and does not prove
which pause setting has the best word or paragraph boundaries.

## Concurrency: capture stays independent, GPU work can compete

Capture uses ordered ingress and does not wait for translation results.
Recognition and translation run in separate model processes. Typical queue
waits remained around 12–26 ms, rather than showing sustained mailbox backlog.
Separate processes nevertheless use the same Apple GPU and memory resources.

In the new run, the 41 partial ASR requests overlapping some translation work
took median **0.865 s**, compared with **0.605 s** for 94 requests without such
overlap. The corresponding p95 values were 1.527 and 0.761 seconds. This is
observational correlation; source complexity and window lengths can also
affect compute time. It supports reducing optional repeated work, not claiming
that translation blocked audio capture.

Final translation has priority over queued previews, but cannot interrupt a
preview already computing. The largest final-MT queue wait was 1.377 seconds.
Graceful Stop completed in 3.402 seconds with no pending source, translation,
queue work or model callbacks.

## Bounded pacing of long-prefix translations

Partial MT now requires append-only growth of at least the configured word
minimum and 25% of the previously translated prefix before promptly
retranslating a long phrase. Small eligible additions can still refresh after
six seconds, preserving progress during slow continuous speech. A larger
configured interval or word minimum remains respected.

The first preview keeps its existing two-second stabilization policy, and
corrections restart stabilization. Queue rejection does not mark text as sent.
The policy keeps complete source context and does not alter final translation
input. Pure policy tests cover the first preview, short prefixes, substantial
growth, bounded refresh, rejected admission, configured minima and corrections.

Optional MT admissions also wait while final ASR is queued, running or awaiting
delivery. Normal partial recognition does not suppress all live MT. Work already
running in another model process is not forcibly interrupted; shared-GPU
contention therefore remains possible even with the reduced admission rate.

## Acoustic recovery and visible uncertainty

A long final that collapses despite substantial preview speech now triggers
one bounded recovery pass inside the existing serialized ASR worker. Conflicting
outro-like previews are withheld without clearing preceding visible text.
This is a consistency guard, not a global ban on thank-you messages.

Recovery reuses the exact prepared float32 audio, forced source language and
transcription task. It partitions the capture into at most four balanced cores,
each at most eight seconds, with up to half a second of audio context on each
side. Balancing avoids a tiny, silent trailing remainder. The two failed clips
would use four 6.16-second cores and three approximately 6.69-second cores.

The MLX Python backend supplies word timestamps for these retries. Word
midpoints determine which core owns each word; repeated spoken words are not
deleted by text similarity. Every window must contain usable aligned speech,
with finite, ordered timestamps inside its captured span and alignment covering
the decoded letters and numbers. Missing, inconsistent or still-collapsed output
invalidates the recovery rather than silently committing a partial candidate.
The pass is bounded to four decodes; it does not recursively retry.

If recovery succeeds, committed translation receives the freshly decoded
result. If it fails or the decoder cannot provide timestamped recovery, no
suspect final or provisional fallback is committed. Previously visible words
remain in a closed **Review needed** card with explicit non-final source and
translation flags. Earlier translations remain provisional, and late ASR/MT
cannot reopen the card. Saved audio, original decoder output, per-window
diagnostics and the unresolved decision remain available for review.

Normal short thanks, meaningful repetition, source-audio ordering, and the
existing silence limits are unchanged. Normal finals use their existing decode;
only consistency failures add the bounded recovery cost. Longer phrases may
preserve useful context, but their last-word-to-final latency and recognition
quality still need validation on matched audio.

## Evidence limits

This run used normal issue-only recording (`debug=false`); full input PCM was
not saved. All 70 saved raw request clips and 70 model inputs passed hashes
and raw-to-float comparisons. Their lack of matching stream segments means
the complete capture cannot be audited or replayed as a continuous stream.
It is not evidence of missing capture samples. MT request text and character
counts were not persisted in timing events, so exact replay of all 21 source
prefix admissions is also unavailable.

The brief `20261108T210010` launch had Hebrew in its parent configuration but
a Turkish source in the saved ASR decision and no MT work. It is excluded from
the Hebrew timing comparison.

Local ignored evidence: `runtime-reports/he-ar-2026-11-09-timing.json` and
`runtime-reports/he-ar-2026-11-09-integrity.json`. These findings do not provide
WER/CER, human-verified translations, or a post-change model benchmark.

## Validation

All **327 Python tests, 20 web tests, 35 embedded pipeline checks and native
protocol self-tests** passed. The regression coverage includes bounded sample
partitioning, overlap ownership, missing/invalid word alignment, genuine
repeats and numbers, decoder failure, no recursive retry, preserved input hashes,
normal short thanks, unresolved-final archival, late-result rejection,
long-prefix pacing and final-ASR priority.

A CPU-only replay of the new consistency trigger against the two saved failure
records catches both, while preserving their original sample counts and hashes
(`runtime-reports/he-ar-recovery-guard-replay.json`). It does not run the model
or establish what the recovered wording will be. Actual recovery quality and
post-change latency still need a new recording or a model replay outside this
test environment. No existing process was restarted. Start a new session and
refresh the web page to load the changes.
