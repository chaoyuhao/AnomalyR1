# AnomalyR1

---

## Notification 🎉

We will be implementing a thorough, more innovative, and more logical comprehensive update covering everything from the code framework to partial design ideas and the paper. Please look forward to its release before **August**.

## Introduction

This project presents an industrial anomaly detection framework built upon the VLM-R1 and Qwen2.5VL-3b multimodal foundation models. It is designed to support visual-language reasoning and anomaly interpretation for industrial imagery, enabling precise and context-aware multimodal queries.

The system is capable of identifying and describing abnormal patterns in complex industrial scenes by leveraging the synergy between visual and textual modalities. It is suitable for applications such as automated inspection, intelligent monitoring, and fault analysis in manufacturing environments.

## Installation

```bash
conda create -n anomaly-r1 python=3.10
conda activate anomaly-r1
bash setup.sh
```

## Training

### GRPO

You can use the following command to train the GRPO model.

```bash
cd open-r1-multimodal
bash run_grpo_ad.sh
```

### SFT

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the SFT model.

Clone the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository and install the dependencies.

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

Download the dataset_info.json, mllm_rec_json.json, and qwen2_5_vl_full_sft.yaml we provided [here](https://huggingface.co/datasets/omlab/VLM-R1/tree/main/sft_related). Put the json files in the `LLaMA-Factory/data` directory and the yaml file in the `LLaMA-Factory/examples/train_full` directory.

Run the following command to train the SFT model.

```bash
llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft.yaml
```

## Dataset

You can download the dataset of MMAD from [here](https://huggingface.co/datasets/jiang-cc/MMAD)

for your customized dataset, please follow the jsonl format below:

```json
{"id": 1, "image": "Clevr_CoGenT_TrainA_R1/data/images/CLEVR_trainA_000001_16885.png", "conversations": [{"from": "human", "value": "<image>What number of purple metallic balls are there?"}, {"from": "gpt", "value": "0"}]}
```

## Evaluation

You can download the checkpoint of GRPO model from [here](https://drive.google.com/drive/folders/1vdicicfz2S4rLfhGFp4iGAncYAsQNexW?usp=drive_link)

And you can use the following command to evaluate the GRPO model.

```bash
mkdir -p MMAD/example/evaluation/AnomalyR1 && cp eval/anomalyr1_query.py MMAD/example/evaluation/AnomalyR1/
cd MMAD/example/evaluation/AnomalyR1
python anomalyr1_query.py --model_path /path/to/your/checkpoint --batch_size 16
```

## Acknowledgement

Thanks to MMAD and VLM-R1, great works in the field of anomaly detection and vision-language models.

## Citation

```bibletex
@misc{chao2025anomalyr1grpobasedendtoendmllm,
      title={AnomalyR1: A GRPO-based End-to-end MLLM for Industrial Anomaly Detection}, 
      author={Yuhao Chao and Jie Liu and Jie Tang and Gangshan Wu},
      year={2025},
      eprint={2504.11914},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.11914}, 
}
```
