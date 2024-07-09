import numpy as np
from scipy.stats import skew, kurtosis

def marginal_distribution_difference(real, generated, bins:int):
    """Calculates the Marginal Distribution Difference (MDD) between real and generated data."""
    batch_size, _, length = real.shape
    mdd = 0
    for i in range(batch_size):
        real_hist, _ = np.histogram(real[i, 0, :], bins=bins, density=True)
        gen_hist, _ = np.histogram(generated[i, 0, :], bins=bins, density=True)
        mdd += np.mean(np.abs(real_hist - gen_hist))
    return mdd / batch_size

def auto_correlation_difference(real, generated):
    """Calculates the Auto-Correlation Difference (ACD) between real and generated data."""
    batch_size, _, length = real.shape
    acd = 0
    for i in range(batch_size):
        real_acf = np.correlate(real[i, 0, :] - np.mean(real[i, 0, :]), real[i, 0, :] - np.mean(real[i, 0, :]), mode='full') / length
        gen_acf = np.correlate(generated[i, 0, :] - np.mean(generated[i, 0, :]), generated[i, 0, :] - np.mean(generated[i, 0, :]), mode='full') / length
        acd += np.mean(np.abs(real_acf - gen_acf))
    return acd / batch_size

def skewness_difference(real, generated):
    """Calculates the Skewness Difference (SD) between real and generated data."""
    batch_size, _, length = real.shape
    sd = 0
    for i in range(batch_size):
        real_skew = skew(real[i, 0, :])
        gen_skew = skew(generated[i, 0, :])
        sd += np.abs(real_skew - gen_skew)
    return sd / batch_size

def kurtosis_difference(real, generated):
    """Calculates the Kurtosis Difference (KD) between real and generated data."""
    batch_size, _, length = real.shape
    kd = 0
    for i in range(batch_size):
        real_kurt = kurtosis(real[i, 0, :])
        gen_kurt = kurtosis(generated[i, 0, :])
        kd += np.abs(real_kurt - gen_kurt)
    return kd / batch_size


if __name__ == '__main__':
    # Example usage:
    # real_data and generated_data should be numpy arrays of shape (batch_size, 1, length)
    real_data = np.random.normal(0, 1, (10, 1, 100))  # example real data
    generated_data = np.random.normal(0, 1, (10, 1, 100))  # example generated data

    print("Marginal Distribution Difference:", marginal_distribution_difference(real_data, generated_data, bins=50))
    print("Auto-Correlation Difference:", auto_correlation_difference(real_data, generated_data))
    print("Skewness Difference:", skewness_difference(real_data, generated_data))
    print("Kurtosis Difference:", kurtosis_difference(real_data, generated_data))
