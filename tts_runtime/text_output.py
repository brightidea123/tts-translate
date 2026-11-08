"""Clean generated model framing before display, logging or speech.

Only known model control tokens are recognized. Ordinary XML, mathematical
comparisons and the original Unicode text are not rewritten. Streaming callers
pass the complete accumulated output with ``final=False``; ambiguous trailing
control-token prefixes are withheld until enough text arrives to classify them.
"""
from __future__ import annotations

import re


# Whisper's multilingual vocabulary also contains language/task/timestamp
# controls. Keep this list local so filtering never imports the MLX runtime.
_WHISPER_LANGUAGES = "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt haw ln ha ba jw su yue".split()
_WHISPER_LITERALS = frozenset({
    *(f"<|{code}|>" for code in _WHISPER_LANGUAGES),
    "<|startoftranscript|>", "<|translate|>", "<|transcribe|>",
    "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
})
_WHISPER_TIMESTAMP = r"<\|(?:[0-2]?\d\.\d{2}|30\.00)\|>"
WHISPER_CONTROL_TOKEN = re.compile(
    "|".join(re.escape(token) for token in sorted(_WHISPER_LITERALS)) + "|" + _WHISPER_TIMESTAMP,
    re.IGNORECASE,
)


_ROLES = ("assistant", "model", "system", "user", "developer", "tool")
_CHANNELS = ("analysis", "thought", "thinking", "reasoning", "final", "answer")
_TERMINALS = frozenset({
    "<end_of_turn>", "<eos>", "</s>", "<|eot_id|>", "<|im_end|>",
    "<|end_of_text|>", "<|return|>", "<|fim_suffix|>",
    "<|endoftext|>",
})
_LITERALS = frozenset({
    *_TERMINALS, *_WHISPER_LITERALS, "<bos>", "<pad>", "<unk>", "<s>",
    "<start_of_turn>", "<|im_start|>", "<|begin_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|finetune_right_pad_id|>",
    "<|start|>", "<|end|>", "<|message|>", "<|channel|>",
    "<|channel>", "<channel|>", "<|meta_sep|>", "<|meta_start|>",
    "<|eom_id|>", "<|recipient|>", "<|endoftext|>",
    "<think>", "</think>", "<thinking>", "</thinking>",
})
_ROLE_NAMES = "|".join(_ROLES)
_CHANNEL_NAMES = "|".join(_CHANNELS)
_TOKEN = re.compile(
    rf"(?P<llama_header><\|start_header_id\|>.*?<\|end_header_id\|>)"
    rf"|(?P<role_header>(?:<start_of_turn>|<\|im_start\|>|<\|start\|>)\s*(?:{_ROLE_NAMES})\b[ \t]*(?:\r?\n)?)"
    rf"|(?P<channel>(?:<\|channel\|?>|<\|meta_sep\|>)[ \t]*(?P<channel_name>{_CHANNEL_NAMES})\b)"
    rf"|(?P<think_open><(?:think|thinking)\b[^>]*>)"
    rf"|(?P<think_close></(?:think|thinking)\s*>)"
    rf"|(?P<whisper_timestamp>{_WHISPER_TIMESTAMP})"
    rf"|(?P<control>{'|'.join(re.escape(token) for token in sorted(_LITERALS, key=len, reverse=True))})",
    re.IGNORECASE | re.DOTALL,
)


def _without_incomplete_tail(text: str, *, final: bool) -> str:
    # A model role is itself streamed. Removing the opener while publishing
    # "ass", "assist", etc. would leak fragments of the header.
    header = re.search(r"(?:<start_of_turn>|<\|im_start\|>|<\|start\|>)([^<\n]*)$", text, re.I)
    if header:
        candidate = header.group(1).strip().casefold()
        if not candidate or any(role.startswith(candidate) for role in _ROLES):
            text = text[:header.start()]
    channel = re.search(r"(?:<\|channel\|?>|<\|meta_sep\|>)[ \t]*([a-z]*)$", text, re.I)
    if channel and (not channel.group(1) or any(name.startswith(channel.group(1).casefold()) for name in _CHANNELS)):
        text = text[:channel.start()]
    opening = text.casefold().rfind("<|start_header_id|>")
    if opening >= 0 and "<|end_header_id|>" not in text[opening:].casefold():
        text = text[:opening]
    unfinished_thought = re.search(r"<(?:think|thinking)\b[^>]*$", text, re.I)
    if unfinished_thought:
        text = text[:unfinished_thought.start()]
    last_angle = text.rfind("<")
    if last_angle >= 0:
        tail = text[last_angle:].casefold()
        if re.fullmatch(r"<\|\d{0,2}(?:\.\d{0,2})?\|?", tail):
            return text[:last_angle]
        if any(token.startswith(tail) and token != tail for token in _LITERALS):
            # At end of generation an isolated '<' can be legitimate maths;
            # '<|...' and longer known prefixes are unambiguous model framing.
            if not final or tail.startswith("<|") or len(tail) >= 3:
                text = text[:last_angle]
    return text


def clean_model_output(text: str, *, final: bool = True) -> str:
    """Return visible generated text, excluding known framing/reasoning.

    The first visible end-of-turn/EOS token ends the answer. This prevents
    repeated stop tokens, or another model turn after them, from reaching TTS.
    ``final=False`` is required for accumulated streaming snapshots; a token
    fragment never needs to be displayed and subsequently retracted.
    """
    if not isinstance(text, str):
        raise TypeError("Generated text must be a string")
    text = _without_incomplete_tail(text, final=final)
    output: list[str] = []
    position, thought_depth = 0, 0
    hidden_channel = False
    separated_hidden_block = False

    def append_visible(piece: str) -> None:
        nonlocal separated_hidden_block
        if not piece:
            return
        if separated_hidden_block and output and output[-1][-1:].isalnum() and piece[:1].isalnum():
            output.append(" ")
        output.append(piece)
        separated_hidden_block = False

    for match in _TOKEN.finditer(text):
        if thought_depth == 0 and not hidden_channel:
            append_visible(text[position:match.start()])
        token = match.group().casefold()
        if match.lastgroup == "think_open":
            thought_depth += 1
        elif match.lastgroup == "think_close":
            if thought_depth == 1:
                separated_hidden_block = True
            thought_depth = max(0, thought_depth - 1)
        elif match.group("channel_name") is not None:
            next_hidden = match.group("channel_name").casefold() not in {"final", "answer"}
            if hidden_channel and not next_hidden:
                separated_hidden_block = True
            hidden_channel = next_hidden
        elif token == "<channel|>":
            if hidden_channel:
                separated_hidden_block = True
            hidden_channel = False
        elif token in _TERMINALS:
            if thought_depth == 0 and not hidden_channel:
                return "".join(output).strip()
            # A completed reasoning message can be followed by a final turn.
            hidden_channel = False
        elif token == "<|end|>":
            if thought_depth == 0 and not hidden_channel and "".join(output).strip():
                return "".join(output).strip()
            hidden_channel = False
        position = match.end()
    if thought_depth == 0 and not hidden_channel:
        append_visible(text[position:])
    return "".join(output).strip()


def _token_id_set(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


def configure_translation_stop_tokens(tokenizer, model_ref: str) -> frozenset[int]:
    """Register Gemma's end-of-turn alongside EOS and return the stop ID set.

    MLX-LM's TokenizerWrapper.add_eos_token() updates the very set consumed by
    stream_generate. Resolve/validate the marker dynamically; never add an
    unknown-token ID as EOS, and never guess global vocabulary numbers.
    Non-Gemma tokenizers keep their own existing stop configuration untouched.
    """
    stops = _token_id_set(getattr(tokenizer, "eos_token_ids", None))
    stops.update(_token_id_set(getattr(tokenizer, "eos_token_id", None)))
    if "gemma" not in str(model_ref).casefold():
        return frozenset(stops)
    marker = "<end_of_turn>"
    token_id = tokenizer.convert_tokens_to_ids(marker)
    unknown_id = getattr(tokenizer, "unk_token_id", None)
    if not isinstance(token_id, int) or token_id == unknown_id:
        raise ValueError(f"Gemma tokenizer does not expose a valid {marker} stop token")
    round_trip = tokenizer.convert_ids_to_tokens(token_id)
    encoded = tokenizer.encode(marker, add_special_tokens=False)
    if round_trip != marker or list(encoded) != [token_id]:
        raise ValueError(f"Gemma tokenizer cannot round-trip the single {marker} stop token")
    add_eos = getattr(tokenizer, "add_eos_token", None)
    if callable(add_eos):
        add_eos(marker)
    elif hasattr(type(tokenizer), "eos_token_ids"):
        tokenizer.eos_token_ids = stops | {token_id}
    stops.add(token_id)
    return frozenset(stops)
