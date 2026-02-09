import torch
import random
import re
from torch.utils.data import Dataset
import pandas as pd


# ============================================================
# Peptide generation dataset
# ============================================================

class PeptideDataset(Dataset):
    """
    Dataset for peptide sequence generation with prompt-based conditioning.
    """

    def __init__(
        self,
        sequences,
        prompts,
        tokenizer,
        max_len=20,
        scheduler=None,
        current_epoch=0,
        ga_dropout=0.0,
        pi_dropout=0.0,
        l_dropout=0.0,
    ):
        self.sequences = sequences
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.scheduler = scheduler
        self.current_epoch = current_epoch

        # Dropout probability for GRAVY attribute
        self.ga_dropout = ga_dropout

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = list(self.sequences[idx])
        full_prompt = self.prompts[idx]

        attrs = full_prompt.split("+")

        # Assume the last three attributes are always pI, Length, and GRAVY
        variable_attrs = attrs[:-3]
        pI_attr, len_attr, gravy_attr = attrs[-3:]

        # --------------------------------------------------
        # GRAVY dropout
        # --------------------------------------------------
        if random.random() < self.ga_dropout:
            fixed_attrs = [pI_attr, len_attr]
        else:
            fixed_attrs = [pI_attr, len_attr, gravy_attr]

        # --------------------------------------------------
        # Curriculum scheduling for variable attributes
        # --------------------------------------------------
        if self.scheduler:
            num_attrs = self.scheduler.get_num_attrs(self.current_epoch)
            num_attrs = min(num_attrs, len(variable_attrs))
            selected_attrs = (
                random.sample(variable_attrs, num_attrs) + fixed_attrs
            )
        else:
            selected_attrs = variable_attrs + fixed_attrs

        prompt = "+".join(selected_attrs)

        # --------------------------------------------------
        # Tokenisation
        # --------------------------------------------------
        seq_ids = self.tokenizer.encode(seq)
        prompt_ids = self.tokenizer.encode_prompt(prompt)
        prompt_len = len(prompt_ids)

        input_ids = prompt_ids + seq_ids

        # Padding / truncation
        pad_id = self.tokenizer.pad_id()
        if len(input_ids) < self.max_len:
            input_ids += [pad_id] * (self.max_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_len]

        attention_mask = [
            1 if token != pad_id else 0 for token in input_ids
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }

    def set_epoch(self, epoch):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch


# ============================================================
# Contrastive learning dataset
# ============================================================

class ContrastDataset(Dataset):
    """
    Dataset for contrastive learning across peptide sequences and prompts,
    supporting multiple tokenisers (custom, BioGPT, and BERT).
    """

    def __init__(
        self,
        sequences,
        prompts,
        tokenizer,
        bert_tokenizer,
        biogpt_tokenizer,
        seq_max_len=20,
        prom_max_len=20,
    ):
        self.sequences = sequences
        self.prompts = prompts

        self.tokenizer = tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.biogpt_tokenizer = biogpt_tokenizer

        self.seq_max_len = seq_max_len
        self.prom_max_len = prom_max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        prompt = self.prompts[idx]

        # Custom tokenizer
        sequence_ids = self.tokenizer.encode(seq)
        prompt_ids = self.tokenizer.encode_prompt(prompt)

        # Preprocessing for protein language models
        sequence = " ".join(seq)
        sequence = re.sub(r"[UZOB]", "X", sequence)

        # BioGPT tokenisation
        biogpt_ids = self.biogpt_tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.prom_max_len,
            return_tensors="pt",
        )

        # BERT tokenisation
        bert_ids = self.bert_tokenizer(
            sequence,
            truncation=True,
            padding="max_length",
            max_length=self.seq_max_len,
            return_tensors="pt",
        )

        # Padding / truncation for custom tokenizer
        pad_id = self.tokenizer.pad_id()

        if len(sequence_ids) < self.seq_max_len:
            sequence_ids += [pad_id] * (self.seq_max_len - len(sequence_ids))
        else:
            sequence_ids = sequence_ids[:self.seq_max_len]

        if len(prompt_ids) < self.prom_max_len:
            prompt_ids += [pad_id] * (self.prom_max_len - len(prompt_ids))
        else:
            prompt_ids = prompt_ids[:self.prom_max_len]

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "sequence_ids": torch.tensor(sequence_ids, dtype=torch.long),
            "biogpt_ids": {
                "input_ids": biogpt_ids["input_ids"].squeeze(0),
                "attention_mask": biogpt_ids["attention_mask"].squeeze(0),
            },
            "bert_ids": {
                "input_ids": bert_ids["input_ids"].squeeze(0),
                "attention_mask": bert_ids["attention_mask"].squeeze(0),
            },
        }


def contrast_collate_fn(batch):
    """
    Collate function for ContrastDataset.
    """
    return {
        "prompt_ids": torch.stack([b["prompt_ids"] for b in batch]),
        "sequence_ids": torch.stack([b["sequence_ids"] for b in batch]),
        "biogpt_ids": {
            "input_ids": torch.stack(
                [b["biogpt_ids"]["input_ids"] for b in batch]
            ),
            "attention_mask": torch.stack(
                [b["biogpt_ids"]["attention_mask"] for b in batch]
            ),
        },
        "bert_ids": {
            "input_ids": torch.stack(
                [b["bert_ids"]["input_ids"] for b in batch]
            ),
            "attention_mask": torch.stack(
                [b["bert_ids"]["attention_mask"] for b in batch]
            ),
        },
    }


# ============================================================
# Aggregation / regression dataset
# ============================================================

class AGRegressionDataset(Dataset):
    """
    Dataset for aggregation-related regression tasks.
    """

    def __init__(self, sequences, tokenizer, max_len=12):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        valid_len = len(seq)

        # Replace uncommon amino acids and add spacing for ESM-style tokenisation
        seq = re.sub(r"[UZOB]", "X", seq)
        spaced_seq = " ".join(list(seq))

        tokens = self.tokenizer(
            spaced_seq,
            truncation=True,
            padding="max_length",
            max_length=self.max_len + 2,
            return_tensors="pt",
        )

        mask = torch.zeros(self.max_len)
        mask[:valid_len] = 1.0

        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "mask": mask,
            "sequence": seq,
        }


# ============================================================
# Peptide dataset with independent attribute dropout
# ============================================================

class PeptideDatasetDropout(Dataset):
    """
    Peptide generation dataset with independent dropout
    for pI, length, and GRAVY attributes.
    """

    def __init__(
        self,
        sequences,
        prompts,
        tokenizer,
        max_len=20,
        scheduler=None,
        current_epoch=0,
        pI_dropout=0.0,
        len_dropout=0.0,
        gravy_dropout=0.0,
    ):
        self.sequences = sequences
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.scheduler = scheduler
        self.current_epoch = current_epoch

        self.pI_dropout = pI_dropout
        self.len_dropout = len_dropout
        self.gravy_dropout = gravy_dropout

    def __len__(self):
        return len(self.sequences)

    def maybe_drop(self, attr, prob):
        """
        Drop a fixed attribute with a given probability.
        """
        return None if random.random() < prob else attr

    def __getitem__(self, idx):
        seq = list(self.sequences[idx])
        full_prompt = self.prompts[idx]

        attrs = full_prompt.split("+")
        variable_attrs = attrs[:-3]
        pI_attr, len_attr, gravy_attr = attrs[-3:]

        # --------------------------------------------------
        # Independent dropout for fixed attributes
        # --------------------------------------------------
        fixed_attrs = [
            a for a in [
                self.maybe_drop(pI_attr, self.pI_dropout),
                self.maybe_drop(len_attr, self.len_dropout),
                self.maybe_drop(gravy_attr, self.gravy_dropout),
            ]
            if a is not None
        ]

        # Ensure at least one fixed attribute is retained
        if not fixed_attrs:
            fixed_attrs = [pI_attr]

        # --------------------------------------------------
        # Curriculum scheduling for variable attributes
        # --------------------------------------------------
        if self.scheduler:
            num_attrs = self.scheduler.get_num_attrs(self.current_epoch)
            num_attrs = min(num_attrs, len(variable_attrs))
            selected_variables = random.sample(variable_attrs, num_attrs)
        else:
            selected_variables = variable_attrs

        prompt = "+".join(selected_variables + fixed_attrs)

        # --------------------------------------------------
        # Tokenisation
        # --------------------------------------------------
        seq_ids = self.tokenizer.encode(seq)
        prompt_ids = self.tokenizer.encode_prompt(prompt)
        prompt_len = len(prompt_ids)

        input_ids = prompt_ids + seq_ids

        pad_id = self.tokenizer.pad_id()
        if len(input_ids) < self.max_len:
            input_ids += [pad_id] * (self.max_len - len(input_ids))
        else:
            input_ids = input_ids[:self.max_len]

        attention_mask = [
            1 if token != pad_id else 0 for token in input_ids
        ]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }

    def set_epoch(self, epoch):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch
