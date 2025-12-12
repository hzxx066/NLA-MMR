# NLA-MMR
This project provides the code implementation of the research work, corresponding to the method NLA-MMR, which can complete model training and inference tasks on the MIMIC-III and MIMIC-IV datasets.

## 1. Environment Setup
Run the following command to install the required dependencies for the project:
```shell
pip install -r requirements.txt
```

## 2. Reproduction Steps
### 1. Data Download and Preprocessing
First, obtain the experimental data (bio_pt embedding) from Google Drive with the download link: https://drive.google.com/file/d/1JbvRD-tODa2SjGdekryCI6DBbYEo9sOR/view?usp=sharing.

After downloading the data, decompress it to the following two directories respectively:
- `./data/mimic-iii/output_atc4`
- `./data/mimic-iv/output_atc4`

### 2. Model Training and Inference
You can start the training and inference process of the NLA-MMR model on the MIMIC-III and MIMIC-IV datasets with one click by executing the script file, and the command is as follows:
```shell
bash run.sh
```

Among them, the core running commands for different datasets are as follows:
1. **MIMIC-III Dataset**
```shell
python main.py \
  --Train \
  --pt_mode bio_pt \
  --max_visit_num 3 \
  --data_file_name ../data/mimic-iii/output_atc4/records_text_iii.pkl \
  --dataset MIMIC-III \
  --med_vocab_size 217
```

2. **MIMIC-IV Dataset**
```shell
python main.py \
  --Train \
  --pt_mode bio_pt \
  --max_visit_num 3 \
  --data_file_name ../data/mimic-iv/output_atc4/records_text_iv.pkl \
  --dataset MIMIC-IV \
  --med_vocab_size 246
```