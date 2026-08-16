"""Unit tests for Phase 2 Foundation Adaptation dataset loader, tokenization, and LoRA setup."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ares.adaptation import (
    DomainSample,
    MultiDomainTextDataset,
    create_multi_domain_dataloader,
    setup_foundation_adapter,
)

try:
    import peft
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class DummyTokenizer:
    """Mock tokenizer for unit testing dataset formatting and labels masking."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, max_length=16, padding="max_length", truncation=True, return_tensors="pt"):
        # Returns synthetic input_ids
        tokens = [2, 3, 4, 5]
        if padding == "max_length":
            tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))

        input_ids = torch.tensor([tokens[:max_length]])
        attention_mask = torch.tensor([[1 if t != self.pad_token_id else 0 for t in tokens[:max_length]]])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DummyConfig(dict):
    """PEFT-compatible config dictionary for dummy testing."""

    def __init__(self, vocab_size: int = 10, hidden_size: int = 8):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            _name_or_path="dummy_model",
            model_type="dummy",
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self._name_or_path = "dummy_model"
        self.model_type = "dummy"

    def to_dict(self):
        return dict(self)


class DummyCausalLM(nn.Module):
    """Mock Causal LM for testing LoRA wrapper initialization."""

    def __init__(self):
        super().__init__()
        self.config = DummyConfig()
        self.q_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 10)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch, seq = input_ids.shape
        x = torch.randn(batch, seq, 8, device=input_ids.device)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = torch.tensor(0.5, device=input_ids.device, requires_grad=True)

        class Outputs:
            pass

        out = Outputs()
        out.logits = logits
        out.loss = loss
        return out


def test_multi_domain_dataset():
    tokenizer = DummyTokenizer()
    samples = [
        DomainSample(sample_id="1", domain="general", text="General knowledge sample"),
        DomainSample(sample_id="2", domain="math", text="Solve 2+2", target="4"),
    ]

    dataset = MultiDomainTextDataset(samples, tokenizer, max_seq_length=8)
    assert len(dataset) == 2

    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert item["input_ids"].shape == (8,)

    # Verify padding tokens are masked with -100 in labels
    pad_mask = item["attention_mask"] == 0
    assert torch.all(item["labels"][pad_mask] == -100)


def test_multi_domain_dataloader():
    tokenizer = DummyTokenizer()
    samples = [
        DomainSample(sample_id=str(i), domain="general", text=f"Sample {i}")
        for i in range(5)
    ]
    dataset = MultiDomainTextDataset(samples, tokenizer, max_seq_length=8)
    dataloader, sampler = create_multi_domain_dataloader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(dataloader))
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)


@pytest.mark.skipif(not PEFT_AVAILABLE, reason="Requires PEFT library")
def test_lora_adapter_setup():
    model = DummyCausalLM()
    peft_model = setup_foundation_adapter(model, r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])

    input_ids = torch.randint(0, 10, (2, 4))
    outputs = peft_model(input_ids=input_ids, labels=input_ids)
    assert outputs.loss is not None
