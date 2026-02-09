# ============================================================
# Standard Libraries
# ============================================================
import os
import pickle

# ============================================================
# Third-party Libraries
# ============================================================
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import cycle
from peft import LoraConfig

# ============================================================
# Project-specific Imports
# ============================================================
from model.gpt_model import PeptideTokenizer, PeptidePromptGPT
from dataset.dataset import PeptideDataset, PeptideDatasetDropout
from common.common import (
    encode_activity_prompts,
    convert_prompt_with_discretized_pI,
    convert_prompt_with_discretized_gravy,
    get_cosine_schedule_with_warmup,
)


# ============================================================
# Prompt Utilities
# ============================================================
def extract_self_assembly_prompt(full_prompt):
    """
    Extract Self-assembly related activity prompt while
    preserving pI, Length and GRAVY attributes.

    Parameters
    ----------
    full_prompt : str
        Original full prompt string.

    Returns
    -------
    str
        Refined prompt string containing only self-assembly
        activity and physicochemical constraints.
    """
    parts = full_prompt.split("+")

    activities = [p.strip() for p in parts if "Self assembly" in p]
    pi_attr = [p for p in parts if p.startswith("pI:")]
    len_attr = [p for p in parts if p.startswith("Length:")]
    gravy_attr = [p for p in parts if p.startswith("GRAVY:")]

    new_prompt = "+".join(activities + pi_attr + len_attr + gravy_attr)
    print(new_prompt)
    return new_prompt


# ============================================================
# Training Utilities
# ============================================================
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    step=0,
):
    """
    Train model for one epoch using standard autoregressive loss.
    """
    model.train()
    total_loss = 0
    tlr_rec = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_lens = batch["prompt_len"]

        # Shift labels for next-token prediction
        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        labels = labels[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()
        fine_tune_scheduler.step()

        total_loss += loss.item()

    if step != 0:
        return total_loss / len(dataloader), step, tlr_rec

    return total_loss / len(dataloader)


def train_one_epoch_mixed(
    model,
    majority_loader,
    minority_loader,
    optimizer,
    criterion,
    device,
    step=0,
    minority_ratio=0.2,
):
    """
    Train one epoch with mixed majority/minority batches.

    minority_ratio controls the proportion of minority samples
    within each training batch.
    """
    model.train()
    total_loss = 0
    tlr_rec = []

    minority_iter = cycle(minority_loader)

    for majority_batch in tqdm(minority_loader):
        batch_size = majority_batch["input_ids"].size(0) * 4
        minority_batch_size = int(batch_size * minority_ratio)
        majority_batch_size = batch_size - minority_batch_size

        majority_batch = {
            k: v[:majority_batch_size]
            for k, v in majority_batch.items()
        }

        minor_batch = next(minority_iter)
        minor_batch = {
            k: v[:minority_batch_size]
            for k, v in minor_batch.items()
        }

        batch = {
            k: torch.cat([majority_batch[k], minor_batch[k]], dim=0).to(device)
            for k in majority_batch.keys()
        }

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lens = batch["prompt_len"]

        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        labels = labels[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()
        fine_tune_scheduler.step()

        total_loss += loss.item()

    return total_loss / len(majority_loader)


def train_one_epoch_mixed_eqbatch(
    model,
    majority_loader,
    minority_loader,
    optimizer,
    criterion,
    device,
    steps_per_epoch=398,
    minority_ratio=0.5,
):
    """
    Mixed training with equalized batch size and fixed steps per epoch.
    """
    model.train()
    total_loss = 0

    majority_iter = cycle(majority_loader)
    minority_iter = cycle(minority_loader)

    for _ in tqdm(range(steps_per_epoch)):
        maj = next(majority_iter)
        minb = next(minority_iter)

        batch_size = maj["input_ids"].size(0) * 2
        minority_batch_size = int(batch_size * minority_ratio)
        majority_batch_size = batch_size - minority_batch_size

        maj = {k: v[:majority_batch_size] for k, v in maj.items()}
        minb = {k: v[:minority_batch_size] for k, v in minb.items()}

        batch = {
            k: torch.cat([maj[k], minb[k]], dim=0).to(device)
            for k in maj.keys()
        }

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        prompt_lens = batch["prompt_len"]

        labels = input_ids.clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        labels = labels[:, 1:]
        input_ids = input_ids[:, :-1]
        attention_mask = attention_mask[:, :-1]

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        fine_tune_scheduler.step()

        total_loss += loss.item()

    return total_loss / steps_per_epoch


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            prompt_lens = batch["prompt_len"]

            labels = input_ids.clone()
            for i, plen in enumerate(prompt_lens):
                labels[i, :plen] = -100

            labels = labels[:, 1:]
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]

            logits = model(input_ids, attention_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# Main Training Pipeline
# ============================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MAX_LEN = 40

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------
    lr = 1e-4
    num_epochs = 20
    batch_size = 1024

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    df = pd.read_csv("./data/dataset_full_prompt_with_gravy_3.csv")
    print(f"Total samples: {len(df)}")

    sequences = df["peptide"].tolist()
    prompts = df["full_prompt"].tolist()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------
    TOKENIZER_PATH = "./result_outputs/trained_models/tokenizer_ga.pkl"
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    print(tokenizer.vocab)

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = PeptidePromptGPT(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=1024,
        tokenizer=tokenizer,
        activity_embed_dict={},
        num_layers=12,
        nhead=16,
        dim_feedforward=512,
        max_len=MAX_LEN,
        distilled_dim=480,
        contract_proj_dim=256,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # --------------------------------------------------------
    # Fine-tuning on minority class: Self Assembly
    # --------------------------------------------------------
    print("\n============================")
    print("🔧 Start fine-tuning on minority class: Self Assembly")
    print("============================\n")

    minority_df = df[df["activity_prompt"].str.contains(
        "Self Assembly", case=False, na=False
    )].copy()

    print(f"Minority samples: {len(minority_df)}")

    minority_df["refined_prompt"] = minority_df["full_prompt"].apply(
        extract_self_assembly_prompt
    )
    minority_df["refined_prompt"] = minority_df["refined_prompt"].apply(
        convert_prompt_with_discretized_gravy
    )
    minority_df["refined_prompt"] = minority_df["refined_prompt"].apply(
        convert_prompt_with_discretized_pI
    )

    minority_sequences = minority_df["peptide"].tolist()
    minority_prompts = minority_df["refined_prompt"].tolist()

    majority_df = df.drop(minority_df.index)
    majority_sequences = majority_df["peptide"].tolist()
    majority_prompts = majority_df["full_prompt"].tolist()

    majority_prompts = [
        convert_prompt_with_discretized_pI(
            convert_prompt_with_discretized_gravy(p)
        )
        for p in majority_prompts
    ]

    majority_dataset = PeptideDatasetDropout(
        majority_sequences,
        majority_prompts,
        tokenizer,
        max_len=MAX_LEN,
        gravy_dropout=0.5,
        pI_dropout=0.2,
        len_dropout=0.2,
    )

    major_loader = DataLoader(
        majority_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
    )

    train_seqs, val_seqs, train_prompts, val_prompts = train_test_split(
        minority_sequences,
        minority_prompts,
        test_size=0.2,
        random_state=42,
    )

    train_dataset = PeptideDatasetDropout(
        train_seqs, train_prompts, tokenizer, max_len=MAX_LEN
    )
    val_dataset = PeptideDataset(
        val_seqs, val_prompts, tokenizer, max_len=MAX_LEN
    )

    minority_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model_path = "./result_outputs/trained_models/best_peptide_gpt_contrast_l12_all2.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))

    fine_tune_lr = 2e-5
    fine_tune_epochs = 20

    fine_tune_optimizer = optim.AdamW(
        model.parameters(), lr=fine_tune_lr, weight_decay=1e-5
    )

    total_steps = fine_tune_epochs * len(minority_loader)
    warmup_steps = int(0.05 * total_steps)
    fine_tune_scheduler = get_cosine_schedule_with_warmup(
        fine_tune_optimizer, warmup_steps, total_steps
    )

    for epoch in range(fine_tune_epochs):
        print(f"\n🔧 Fine-tune Epoch {epoch + 1}/{fine_tune_epochs}")

        train_loss = train_one_epoch_mixed_eqbatch(
            model,
            major_loader,
            minority_loader,
            fine_tune_optimizer,
            criterion,
            device,
            steps_per_epoch=35,
            minority_ratio=0.5,
        )

        val_loss = validate_one_epoch(model, val_loader, criterion, device)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    final_path = (
        "./result_outputs/trained_models/"
        "model_finetune_selfassembly_final_l12_al_2-3.pt"
    )
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Fine-tuned model saved to: {final_path}")
    print("\n🎉 Fine-tuning complete!")
