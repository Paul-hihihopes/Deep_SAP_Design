import torch
from torch.utils.data import Dataset
from itertools import groupby
from tqdm import tqdm

from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import (
    to_pdb,
    Protein as OFProtein
)
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37


# ============================================================
# Global configuration
# ============================================================

# Enable TensorCore acceleration for better performance
torch.backends.cuda.matmul.allow_tf32 = True

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Reserved file names on Windows systems
WINDOWS_RESERVED_NAMES = [
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5",
    "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4",
    "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
]


# ============================================================
# Dataset definition
# ============================================================

class SequenceDataset(Dataset):
    """
    A simple PyTorch Dataset wrapper for protein sequences.
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


# ============================================================
# Utility functions
# ============================================================

def convert_outputs_to_pdb(outputs):
    """
    Convert ESMFold model outputs to PDB format strings.

    Parameters
    ----------
    outputs : dict
        Raw outputs from EsmForProteinFolding.

    Returns
    -------
    list[str]
        A list of PDB-formatted strings.
    """
    # Convert atom14 positions to atom37 representation
    final_atom_positions = atom14_to_atom37(
        outputs["positions"][-1],
        outputs
    )

    # Move tensors to CPU and convert to NumPy
    outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().detach().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]

    pdbs = []

    for i in range(outputs["aatype"].shape[0]):
        pred_protein = OFProtein(
            aatype=outputs["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=outputs["residue_index"][i] + 1,
            b_factors=outputs["plddt"][i],
            chain_index=(
                outputs["chain_index"][i]
                if "chain_index" in outputs else None
            ),
        )
        pdbs.append(to_pdb(pred_protein))

    return pdbs


def batch_by_length(sequences, batch_size=64):
    """
    Group sequences into batches by identical length.

    This avoids padding and reduces unnecessary computation.

    Parameters
    ----------
    sequences : list[str]
        Protein sequences.
    batch_size : int
        Maximum batch size for each length group.

    Returns
    -------
    list[list[str]]
        A list of sequence batches.
    """
    sequences.sort(key=len)
    batches = []

    for _, group in groupby(sequences, key=len):
        group = list(group)
        for i in range(0, len(group), batch_size):
            batches.append(group[i:i + batch_size])

    return batches


# ============================================================
# Main inference pipeline
# ============================================================

def get_pdbs(sequences, output_dir="./data/pdb_buf/"):
    """
    Predict protein structures using ESMFold and save them as PDB files.

    Parameters
    ----------
    sequences : list[str]
        Input protein sequences.
    output_dir : str
        Directory to save generated PDB files.
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "./pLM/esmfold_v1",
        local_files_only=True
    )

    # Load ESMFold model
    model = EsmForProteinFolding.from_pretrained(
        "./pLM/esmfold_v1",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    ).to(device)

    # Fine-grained precision and memory control
    model.esm = model.esm.half()
    model.trunk.set_chunk_size(16)

    # Batch sequences by length
    batches = batch_by_length(sequences, batch_size=32)

    # Sanity check: ensure equal length within each batch
    for batch in batches:
        tokenized = tokenizer(batch)
        lengths = [len(x) for x in tokenized["input_ids"]]
        print(set(lengths))

    # Run structure prediction
    for batch_sequences in tqdm(batches):
        tokenized_input = tokenizer(
            batch_sequences,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            truncation=True
        )["input_ids"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(tokenized_input)

        pdb_strings = convert_outputs_to_pdb(outputs)

        for seq, pdb in zip(batch_sequences, pdb_strings):
            filename = seq
            if seq.upper() in WINDOWS_RESERVED_NAMES:
                filename = f"{seq}_renamed"

            with open(f"{output_dir}/{filename}.pdb", "w") as f:
                f.write(pdb)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    get_pdbs(["GIILNVLNSH"])
