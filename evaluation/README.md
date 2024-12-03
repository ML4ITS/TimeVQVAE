# Details of Logged Metrics


## Stage1

* `recons_loss`: reconstruction loss
    * `LF`: low-frequency
    * `HF`: low-frequency
* `perlexity`: codebook usage; high perpexlity corresponds to high usage of codes in a codebook.




## Stage2

* runing_metrics
    * `FID`: FID score (lower, the better)
    * `MDD`: Marginal Distribution Difference (lower, the better) [1]
    * `ACD`: AutoCorrelation Difference (lower, the better) [1]
    * `SD`: Skewness Difference (lower, the better) [1]
    * `KD`: Kurtosis Difference (lower, the better) [1]

* train/val
    * `mask_pred_loss_l`: prior loss for the LF prior model
    * `mask_pred_loss_h`: prior loss for the HF prior model

* Media
    * `PCA on Z (['Z_test', 'Zhat'])`: comparison between a test set and a generated set in an evaluation latent space. The current default setting is using a ROCKET encoder to project data into the evaluation latent space. (in the paper, a pretrained FCN encoder is used but we found ROCKET give better representations, i.e., unbiased latent space.)
    * `visual comp (X_test vs Xhat)`: comparison between $X$ and $\hat{X}$ in a data space.


## Evaluation

* `PCA on Z (['Z_test', 'Z_rec_test'])`: comparison between a test set and a set of reconstructed samples in the evaluation latent space. `rec` denotes reconstruction.
* `X_test_c`: test time series for different classes. `cls_idx` denotes a class index.
* `Xhat_c`: class-conditionally generated time series.




## References
[1] Ang, Yihao, et al. "TSGBench: Time Series Generation Benchmark." Proceedings of the VLDB Endowment 17.3 (2023): 305-318.