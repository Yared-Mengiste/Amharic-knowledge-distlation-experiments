"""Exact native-template assistant spans, including the turn-ending token."""
import hashlib

from students.config import SETTINGS

ENDINGS = {"llama": "<|eot_id|>", "gemma": "<turn|>"}


def template_kwargs():
    # Llama's template otherwise injects today's date, changing preflight/resume/eval.
    return {"enable_thinking": False, "date_string": SETTINGS["date_string"]}


def template_hash(tokenizer):
    return hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest()


def render_prompt(tokenizer, messages):
    if [m["role"] for m in messages] != ["user"]:
        raise ValueError("Only the existing one-user-turn schema is supported")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs())


def encode_conversation(tokenizer, row, model):
    messages = row["messages"]
    if [m["role"] for m in messages] != ["user", "assistant"]:
        raise ValueError(f"{row['id']}: only user/assistant pairs are supported")
    for message in messages:
        if any(token in message["content"] for token in tokenizer.all_special_tokens):
            raise ValueError(f"{row['id']}: message contains a reserved chat token")
    prefix = render_prompt(tokenizer, messages[:-1])
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **template_kwargs())
    if not rendered.startswith(prefix):
        raise ValueError(f"{row['id']}: native template prefix differs for training and generation")
    answer = messages[-1]["content"].strip()
    expected = prefix + answer + ENDINGS[model]
    if not rendered.startswith(expected) or rendered[len(expected):].strip():
        raise ValueError(f"{row['id']}: template rewrote answer or changed ending; inspect the pinned template")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start, end = len(prefix), len(expected)
    labels = [-100] * len(ids)
    for i, (left, right) in enumerate(offsets):
        if left < start < right or left < end < right:
            raise ValueError(f"{row['id']}: token crosses assistant span boundary")
        if start <= left < right <= end:
            labels[i] = ids[i]
    ending_id = tokenizer.convert_tokens_to_ids(ENDINGS[model])
    supervised = [token for token in labels if token != -100]
    if len(supervised) < 2 or supervised[-1] != ending_id:
        raise ValueError(f"{row['id']}: answer/ending is not supervised")
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def padded_features(features, pad_id):
    """Right padding masks by POSITION, even when padding and EOS share an ID."""
    length = max(len(f["input_ids"]) for f in features)
    result = {key: [] for key in ("input_ids", "attention_mask", "labels")}
    for feature in features:
        padding = length - len(feature["input_ids"])
        for key, value in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100)):
            result[key].append(feature[key] + [value] * padding)
    return result


class AssistantCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        import torch
        return {k: torch.tensor(v, dtype=torch.long) for k, v in padded_features(features, self.pad_id).items()}
