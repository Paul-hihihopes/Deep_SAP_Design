import torch
import torch.nn as nn
from common.common import top_k_logits


# ============================================================
# Tokenizer
# ============================================================
class PeptideTokenizer:
    """
    Tokenizer for peptide-related discrete tokens, including
    amino acids, activities, physicochemical properties, etc.
    """

    def __init__(self, token_dict, add_pad=True, add_unk=True):
        """
        Parameters
        ----------
        token_dict : dict[str, list[str]]
            Mapping from category name to a list of tokens.
            Example:
            {
                "AA": ["A", "C", "D"],
                "Activity": ["Activator"],
                "pI": ["pI:3.5-4.0"],
                "Length": ["Length:14"]
            }
        add_pad : bool
            Whether to add a PAD token.
        add_unk : bool
            Whether to add an UNK token.
        """
        self.vocab = {}
        self.idx2token = {}

        idx = 0
        for _, tokens in token_dict.items():
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1

        # Special tokens
        if add_pad:
            self.pad_token = "[PAD]"
            self.vocab[self.pad_token] = idx
            idx += 1
        else:
            self.pad_token = None

        if add_unk:
            self.unk_token = "[UNK]"
            self.vocab[self.unk_token] = idx
            idx += 1
        else:
            self.unk_token = None

        self.idx2token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, tokens):
        """
        Encode a list of tokens into token indices.
        """
        return [
            self.vocab.get(t, self.vocab.get(self.unk_token))
            for t in tokens
        ]

    def encode_prompt(self, prompt_string, sep="+"):
        """
        Encode a prompt string separated by a delimiter.
        """
        tokens = prompt_string.split(sep)
        return self.encode(tokens)

    def decode(self, indices):
        """
        Decode token indices back to tokens.
        """
        return [
            self.idx2token.get(i, self.unk_token)
            for i in indices
        ]

    def pad_id(self):
        """
        Return the index of the PAD token.
        """
        return self.vocab.get(self.pad_token, 0)

    def batch_encode_prompts(self, prompt_list, sep="+"):
        """
        Encode a batch of prompt strings.
        """
        return [self.encode_prompt(p, sep) for p in prompt_list]


# ============================================================
# GPT-style Transformer Model for Peptide Prompting
# ============================================================
class PeptidePromptGPT(nn.Module):
    """
    GPT-style Transformer model for peptide prompt-based generation
    and representation learning.
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        tokenizer,
        activity_embed_dict,
        num_layers=4,
        nhead=8,
        dim_feedforward=512,
        max_len=64,
        distilled_dim=None,
        contract_proj_dim=512,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dim_feedforward = dim_feedforward
        self.distilled_dim = distilled_dim

        # ----------------------------------------------------
        # Embedding layers
        # ----------------------------------------------------
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.token_embedding.weight.data, -0.1, 0.1)

        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.embedding_dropout = nn.Dropout(0.3)

        # ----------------------------------------------------
        # Transformer encoder
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Initialize prompt embeddings (e.g., activity tokens)
        self._init_prompt_embedding(activity_embed_dict)

        # Optional distillation projection
        if distilled_dim is not None:
            self.distill_projection = nn.Linear(
                embedding_dim, distilled_dim
            )

        # Optional contrastive projection head
        if contract_proj_dim is not None:
            self.contract_projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, contract_proj_dim),
            )

    # ========================================================
    # Internal utilities
    # ========================================================
    def _init_prompt_embedding(self, activity_embed_dict):
        """
        Initialize token embeddings for specific prompt tokens
        (e.g., activity labels) using predefined embeddings.
        """
        if activity_embed_dict is None:
            return

        for activity, emb in activity_embed_dict.items():
            idx = self.tokenizer.vocab.get(activity)
            if idx is not None:
                self.token_embedding.weight.data[idx] = torch.tensor(emb)

    # ========================================================
    # Encoding utilities
    # ========================================================
    def encode(self, input_ids, mode="norm"):
        """
        Encode a sequence into a pooled embedding.

        Parameters
        ----------
        input_ids : Tensor [B, L]
            Token indices.
        mode : str
            - "norm": normal pooled embedding
            - "distilled": apply distillation projection
            - "contrast": apply contrastive projection
        """
        batch_size, seq_len = input_ids.size()

        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(pos_ids)

        emb = self.embedding_dropout(token_emb + pos_emb)
        out = self.transformer(emb)  # [B, L, H]

        pooled = out.mean(dim=1)  # Mean pooling

        if (
            self.distilled_dim is not None
            and self.distilled_dim != self.embedding_dim
            and mode == "distilled"
        ):
            pooled = self.distill_projection(pooled)

        if mode == "contrast":
            pooled = self.contract_projection(pooled)

        return pooled

    # ========================================================
    # Forward (autoregressive training)
    # ========================================================
    def forward(self, input_ids, attention_mask=None, inputs_embeds=None, **kwargs):
        """
        Forward pass for autoregressive language modeling.
        """
        batch_size, seq_len = input_ids.size()

        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(
            seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(pos_ids)

        emb = self.embedding_dropout(token_emb + pos_emb)

        # Causal mask for GPT-style autoregression
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float("-inf"),
            diagonal=1,
        )

        out = self.transformer(emb, mask=causal_mask.isinf())
        logits = self.fc_out(out)

        return logits

    # ========================================================
    # Text generation
    # ========================================================
    def generate(
        self,
        prompt_tokens,
        num_samples=5,
        max_gen_len=20,
        temperature=1.0,
        top_k=10,
    ):
        """
        Generate sequences from a given prompt using top-k sampling.
        """
        self.eval()
        device = next(self.parameters()).device

        prompt_batch = [prompt_tokens] * num_samples
        input_ids = torch.tensor(
            prompt_batch, dtype=torch.long, device=device
        )

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_gen_len):
                seq_len = generated.size(1)

                pos_ids = torch.arange(
                    seq_len, device=device
                ).unsqueeze(0).expand(num_samples, seq_len)

                token_emb = self.token_embedding(generated)
                pos_emb = self.position_embedding(pos_ids)
                emb = self.embedding_dropout(token_emb + pos_emb)

                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=device) * float("-inf"),
                    diagonal=1,
                )

                out = self.transformer(emb, mask=causal_mask.isinf())
                logits = self.fc_out(out)

                next_token_logits = logits[:, -1, :] / temperature
                filtered_logits = top_k_logits(next_token_logits, top_k)

                probs = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

        gen_ids = generated[:, input_ids.size(1):].tolist()
        gen_tokens = [self.tokenizer.decode(ids) for ids in gen_ids]

        return gen_tokens
