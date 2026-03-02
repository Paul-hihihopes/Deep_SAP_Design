Code for the deep learning–based self-assembling peptide design framework proposed in our paper "DeepSAP: An Integrated Generative and Predictive Deep-Learning Framework for Controllable Design of Self-Assembling Peptides".

<pre> project_root/
 ├── common/
 ├── model/
 │    ├── ap_model.py
 │    ├── gpt_model.py
 ├── data/
 ├── dataset/
 │    ├── dataset.py
 ├── pLM/
 │    ├── esm2/
 │    └── biogpt/
 ├── result_outputs/
 ├── requirements.txt
 ├── model_training.py
 ├── brf_predict.py
 ├── prompt_finetune.py
 ├── model_generate.py
 ├── .gitignore
 └── README.md</pre>
 
 Before running the code, please ensure:
 
  1. The folder `pLM/` exists in the project root directory.
  2. ESM2 weights are placed under `pLM/esm2/`.
  4. BioGPT weights are placed under `pLM/biogpt/`.

## Environment
Tested on:
- Python 3.12
- PyTorch 2.5.0
- CUDA 12.8

## Installation
<pre>git clone https://github.com/Paul-hihihopes/Deep_SAP_Design.git

cd Deep_SAP_Design

pip install -r requirements.txt</pre>

## Model Checkpoints and Data

Due to file size limitations, the trained model checkpoints and other files are not included in this repository.

They can be downloaded from:

https://pan.baidu.com/s/18sdYlMLot79H-oOoVy85nQ?pwd=m6cr

## Predictor Usage

The testing script for the predictor model is located in:
<pre>
 common/xg_features.py
</pre>

To run tesing:
<pre>
cd common
python xg_features.py 
</pre>

The prediction script for the trained BRF classifier is:

<pre> brf_predict.py </pre>

Run Prediction
<pre> python brf_predict.py \ --fasta seqs_example.fasta \ --top_k 20 \ --output ./result_outputs/generated_analysis/prediction_results.csv </pre>

Explanation of arguments:
| Argument   | Description                                   |
| ---------- | --------------------------------------------- |
| `--fasta`  | Input FASTA file containing peptide sequences |
| `--top_k`  | Number of sequences selected by MMR           |
| `--output` | Path to save prediction results (CSV)         |

## Generator Usage
The script for sequences generation is:
<pre>
model_generate.py
</pre>

To generate sequences with custom property constraints:
<pre>
python model_generate.py \
    --prompt "Self assembly+pI:6.0-6.5+Length:5+GRAVY:0.0-0.5" \
    --num_samples 500 \
    --max_len 10 \
    --output_dir ./outputs/
</pre>

Explanation of arguments:
| Argument        | Description                             |
| --------------- | --------------------------------------- |
| `--prompt`      | Property-controlled generation prompt   |
| `--num_samples` | Number of sequences to generate         |
| `--max_len`     | Maximum peptide length                  |
| `--output_dir`  | Directory to save generated FASTA files |


If no --prompt argument is provided, the script automatically runs the predefined benchmark tasks used in the manuscript:
<pre>
python model_generate.py
</pre>


