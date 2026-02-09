import re
import torch
import torch.nn as nn
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
)
from tqdm import tqdm
from transformers import (
    BertModel,
    BertConfig,
    EsmModel,
)


# ============================================================
# Enhanced ProtBERT + BiLSTM + Attention regressor
# ============================================================

class EnhancedPBertLSTM_UsingConf(nn.Module):
    """
    Enhanced ProtBERT-based model with BiLSTM and self-attention
    for sequence-level regression.
    """

    def __init__(
        self,
        hidden_size=1024,
        lstm_hidden_size=512,
        lstm_layers=2,
        max_length=10,
        unfreeze_bert_layers=4,
        bert_path="./prot_bert",
        return_features=None,
        bert_config=None,
    ):
        super().__init__()

        # --------------------------------------------------
        # BERT configuration
        # --------------------------------------------------
        if bert_config is None:
            bert_config = BertConfig.from_pretrained(
                bert_path,
                output_hidden_states=True,
            )
        else:
            # Ensure num_labels exists in config
            bert_config.num_labels = 2

        # Load ProtBERT model
        self.bert_model = BertModel.from_pretrained(
            bert_path,
            config=bert_config,
        )

        # --------------------------------------------------
        # BiLSTM encoder
        # --------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(0.1)

        # --------------------------------------------------
        # Self-attention layer
        # --------------------------------------------------
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # --------------------------------------------------
        # Deep regression head
        # --------------------------------------------------
        self.regression_head = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.return_features = return_features

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (B, L)
        attention_mask : torch.Tensor
            Attention mask of shape (B, L)

        Returns
        -------
        torch.Tensor or tuple
            Regression scores, optionally with intermediate features.
        """
        # --------------------------------------------------
        # ProtBERT forward
        # --------------------------------------------------
        self.bert_model.train()
        bert_out = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, L, D)

        # --------------------------------------------------
        # BiLSTM encoding
        # --------------------------------------------------
        lstm_out, _ = self.lstm(bert_out)  # (B, L, 2H)

        # --------------------------------------------------
        # Self-attention
        # --------------------------------------------------
        attn_out, _ = self.attention(
            lstm_out,
            lstm_out,
            lstm_out,
        )
        attn_out = self.lstm_dropout(attn_out)

        # --------------------------------------------------
        # Optional feature return
        # --------------------------------------------------
        if self.return_features == "lstm":
            last_lstm_feature = lstm_out[:, -1, :]
            preds = self.regression_head(last_lstm_feature).squeeze(-1)
            return preds, last_lstm_feature

        # --------------------------------------------------
        # Regression prediction (per position)
        # --------------------------------------------------
        return self.regression_head(attn_out).squeeze(-1)


# ============================================================
# Inference utilities
# ============================================================

def predict(
    model,
    tokenizer,
    sequences,
    max_length=12,
    batch_size=128,
):
    """
    Predict sequence-level scores by summing position-wise outputs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Preprocess sequences
    seqs = [
        " ".join(re.sub(r"[UZOB]", "X", seq))
        for seq in sequences
    ]

    encoded = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_predictions = []

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            preds = model(input_ids, attention_mask)  # (B, L)

            # Valid positions: exclude special tokens and padding
            valid_mask = (
                (input_ids != 0)
                & (input_ids != 2)
                & (input_ids != 3)
            )

            for i in range(input_ids.size(0)):
                valid_len = valid_mask[i].sum().item()
                score = preds[i, 1 : valid_len + 1].sum().item()
                all_predictions.append(score)

    return all_predictions


def predict_with_positions(
    model,
    tokenizer,
    sequences,
    max_length=12,
    batch_size=64,
):
    """
    Predict both total scores and per-position scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    seqs = [
        " ".join(re.sub(r"[UZOB]", "X", seq))
        for seq in sequences
    ]

    encoded = tokenizer(
        seqs,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_scores = []
    position_scores = []

    with torch.no_grad():
        for input_ids, attention_mask in tqdm(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            preds = model(input_ids, attention_mask)

            valid_mask = (
                (input_ids != 0)
                & (input_ids != 2)
                & (input_ids != 3)
            )

            for i in range(input_ids.size(0)):
                valid_len = valid_mask[i].sum().item()
                pos_scores = preds[i, 1 : valid_len + 1].tolist()
                total_scores.append(sum(pos_scores))
                position_scores.append(pos_scores)

    return total_scores, position_scores


# ============================================================
# ESM positional regressor
# ============================================================

class ESMPositionalRegressor(nn.Module):
    """
    ESM-based positional regressor producing one scalar per residue.
    """

    def __init__(self, model_name="./esm2", freeze_esm=False):
        super().__init__()

        self.esm = EsmModel.from_pretrained(model_name)
        if freeze_esm:
            self.esm.requires_grad_(False)

        esm_dim = self.esm.config.hidden_size
        self.head = nn.Linear(esm_dim, 1)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass.

        Returns
        -------
        torch.Tensor
            Position-wise predictions of shape (B, L)
        """
        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        embeddings = outputs.last_hidden_state  # (B, L+2, D)
        embeddings = embeddings[:, 1:-1, :]     # remove [CLS] and [EOS]

        preds = self.head(embeddings).squeeze(-1)
        return preds
