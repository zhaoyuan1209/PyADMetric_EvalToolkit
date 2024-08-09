PyADMetric_EvalToolkit (PyAD_Metric): A Python-based Simple yet Efficient Evaluation Toolbox for Segmentation-like tasks.

## 2D Anomaly Detection

**AUROC**: Area Under the Receiver Operating Characteristic Curve

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\text{AUROC}=\int_{0}^{1}\text{TPR(FPR)},\d(\text{FPR})" alt="AUROC formula" />
</p>

**AUPR**: Area Under the Precision-Recall Curve

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\text{AUPR}=\int_{0}^{1}P(R),\d(\text{R})" alt="AUPR formula" />
</p>

**AP**: Average Precision

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\text{AP}=\sum_{n}(R_n-R_{n-1})P_n" alt="AP formula" />
</p>

**PRO**: Per-Region Overlap is defined as the average relative overlap of the binary prediction *P* with each connected component Ck​ of the ground truth.

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\text{PRO}=\frac{1}{K}\sum_{k=1}^{K}\frac{|P\cap&space;C_k|}{|C_k|}" alt="PRO formula" />
</p>

**F1max**: F1-score-max (F1-max) -- F1-score at optimal threshold *θ* for a clearer view against potential data imbalance

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\text{F1}_{\text{max}}(\theta)=\max_{\theta}\left(\frac{2&space;\times&space;\text{Precision}(\theta)&space;\times&space;\text{Recall}(\theta)}{\text{Precision}(\theta)+\text{Recall}(\theta)}\right)" alt="F1max formula" />
</p>

## 3D Anomaly Detection Continue......

### 评测指标参考文献

```text
@article{bergmann2021mvtec,
  title={The mvtec 3d-ad dataset for unsupervised 3d anomaly detection and localization},
  author={Bergmann, Paul and Jin, Xin and Sattlegger, David and Steger, Carsten},
  journal={arXiv preprint arXiv:2112.09045},
  year={2021}
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
