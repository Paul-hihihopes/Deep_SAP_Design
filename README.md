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
 ├── xg_predict.py
 ├── prompt_finetune.py
 ├── generate_seq_analysis.py
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

cd root

pip install -r requirements.txt</pre>

## Model Checkpoints and Data

Due to file size limitations, the trained model checkpoints and processed datasets are not included in this repository.

They can be downloaded from:

https://pan.baidu.com/s/18sdYlMLot79H-oOoVy85nQ?pwd=m6cr

