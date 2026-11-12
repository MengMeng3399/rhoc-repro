import hashlib
import re

import torch
from torch.nn import functional as F

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_TOKEN = re.compile(r"[\u3400-\u9fff]|[^\W_]+", re.UNICODE)


def mask_requirements(requirements, api_rows, dataset):
    fields = ("name", "category", "tag") if dataset == "pw" else ("name", "category")
    terms = set()
    for row in api_rows:
        for field in fields:
            value = row.get(field, [])
            values = [value] if isinstance(value, str) else value
            terms.update(term.strip() for term in values or [] if term.strip())

    patterns = []
    for term in sorted(terms, key=lambda value: (-len(value), value.casefold(), value)):
        literal = re.escape(term)
        left = r"(?<!\w)" if term[0].isalnum() and not _CJK.match(term[0]) else ""
        right = r"(?!\w)" if term[-1].isalnum() and not _CJK.match(term[-1]) else ""
        patterns.append(f"{left}{literal}{right}")

    if not patterns:
        return list(requirements)
    pattern = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return [pattern.sub("[MASK]", text) for text in requirements]


class BGEEncoder:
    def __init__(self, checkpoint, device, width):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        if self.model.config.hidden_size != width:
            raise ValueError(f"{checkpoint} has an unexpected hidden size")

    @torch.inference_mode()
    def encode(self, texts, batch_size):
        batches = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            result = self.model(**batch).last_hidden_state[:, 0, :]
            batches.append(F.normalize(result.float(), dim=-1).cpu())
        return torch.cat(batches)


class DemoEncoder:
    def __init__(self, width):
        self.width = width

    def encode(self, texts, batch_size):
        result = torch.zeros(len(texts), self.width)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.casefold()):
                digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "little") % self.width
                result[row, index] += 1 if digest[-1] & 1 else -1
        return F.normalize(result, dim=-1)
