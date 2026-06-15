# AI CUP 2026 - Table Tennis Rally Prediction

## Introduction

本專案為 AI CUP 2026「基於時序資料之桌球戰術與結果預測競賽」之程式碼。

本研究以 LSTM 為核心模型，利用桌球 Rally（回合）的時間序列資訊，預測：

* actionId
* pointId
* serverGetPoint

最終提交結果採用多個不同 Random Seed 訓練之模型，再透過 Voting Ensemble 產生最終預測結果。

---

# Environment

建議執行環境：

* Windows 10
* Python 3.14.4

主要使用套件：

* PyTorch
* NumPy
* Pandas
* Scikit-learn

請先安裝所需套件：

```bash
pip install -r requirements.txt
```

---

# Dataset

請將主辦單位提供之資料放置於專案資料夾：

```
train.csv
test_new.csv
sample_submission.csv
```

---

# Training

請分別利用不同 Random Seed 訓練模型並產生預測結果。

例如：

## Seed 3407

```bash
python merge_laststep_fixed_submit.py --train train.csv --test test_new.csv --sample sample_submission.csv --out submission_h256_k5_seed3407.csv --seed 3407
```

---

## Seed 3408

```bash
python merge_laststep_fixed_submit.py --train train.csv --test test_new.csv --sample sample_submission.csv --out submission_h256_k5_seed3408.csv --seed 3408
```

---

## Seed 777

```bash
python merge_laststep_fixed_submit.py --train train.csv --test test_new.csv --sample sample_submission.csv --out submission_h256_k5_seed777.csv --seed 777
```

---

## Base Model

```bash
python merge_laststep_fixed_submit.py --train train.csv --test test_new.csv --sample sample_submission.csv --out submission_h256_k5.csv
```

完成後應產生四個檔案：

```
submission_h256_k5_seed3407.csv

submission_h256_k5_seed3408.csv

submission_h256_k5_seed777.csv

submission_h256_k5.csv
```

---

# Voting Ensemble

請執行：

```bash
python ensemble_3407_3408_777_base.py
```

程式會讀取：

* submission_h256_k5_seed3407.csv
* submission_h256_k5_seed3408.csv
* submission_h256_k5_seed777.csv
* submission_h256_k5.csv

並進行 Voting Ensemble。

其中：

* serverGetPoint 採用 Seed3407 模型輸出。
* actionId 與 pointId 採用 3407、3408、777 及 Base 四個模型共同投票決定最終結果。

---

# Output

程式完成後將輸出：

```
submission_vote_3407_3408_777_base.csv
```

此檔案即為本研究最終提交之預測結果。

---

# Files

本專案主要包含：

## merge_laststep_fixed_submit.py

功能：

* 資料前處理
* 建立時間序列資料
* LSTM 模型訓練
* Prediction
* 產生 Submission 檔案

---

## ensemble_3407_3408_777_base.py

功能：

* 讀取四個模型預測結果
* Voting Ensemble
* 輸出最終 Submission

---

# Notes

本研究曾另外測試 Transformer、3-fold Cross Validation 與 5-fold Cross Validation 等方法作為模型比較，但最終提交版本採用 LSTM 結合 Multi-Seed Voting Ensemble，作為最佳預測方案。
