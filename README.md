# PyADMetric_EvalToolkit

PyADMetric_EvalToolkit (PyAD_Metric): A Python-based Simple yet Efficient Evaluation Toolbox for Segmentation-like tasks

## 2D Anomaly Detection

### AUROC: Area Under the Receiver Operating Characteristic Curve

$ \text{AUROC} = \int_{0}^{1} TPR(FPR) \, d(\text{FPR}) $

### AUPR: Area Under the Precision-Recall Curve

$ \text{AUPR} = \int_{0}^{1} P(R) \, d(\text{R})$

## AP: Average Precision

$ \text{AP} = \sum_n (R_n - R_{n-1}) P_n$

### PRO: Per-Region Overlap is defined as the average relative overlap of the binary prediction $P$ with each connected component $C_k$ of the ground truth.

$ \text{PRO} = \frac{1}{K} \sum_{k=1}^{K} \frac{|P \cap C_k|}{|C_k|}$

### F1max:F1-score-max (F1-max)-- $F_1$-score at optimal threshold $\theta$ for a clearer view against potential data imbalance

$ \text{F1}_{\text{max}} = \max_{\theta} \left( \frac{2 \times \text{Precision}(\theta) \times \text{Recall}(\theta)}{\text{Precision}(\theta) + \text{Recall}(\theta)} \right) $

### ## 3D Anomaly Detection

## 评测指标参考文献

```text
@inproceedings{Fmax_mean,
title={Frequency-tuned salient region detection},
author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
booktitle= CVPR,
pages={1597--1604},
year={2009}
}  
@inproceedings{zou2022spot,
  title={Spot-the-difference self-supervised pre-training for anomaly detection and segmentation},
  author={Zou, Yang and Jeong, Jongheon and Pemula, Latha and Zhang, Dongqing and Dabeer, Onkar},
  booktitle={European Conference on Computer Vision},
  pages={392--408},
  year={2022},
  organization={Springer}
}
```
