import numpy as np
from scipy.stats import skew, kurtosis, gaussian_kde

def marginal_distribution_difference(real, generated):
    """
    Calculates the Marginal Distribution Difference (MDD) between real and generated data.
    
    NB!
    - The original code from the TSBBench paper uses histogram differences, which may be sensitive to the choice of bins. 
      To improve this, we consider using kernel density estimation (KDE) for a smoother and more robust comparison.
    """
    real_values = real.reshape(-1)
    generated_values = generated.reshape(-1)
    
    real_kde = gaussian_kde(real_values)
    gen_kde = gaussian_kde(generated_values)
    
    x = np.linspace(min(real_values.min(), generated_values.min()), 
                    max(real_values.max(), generated_values.max()), 100)
    
    mdd = np.mean(np.abs(real_kde(x) - gen_kde(x)))
    return mdd

def auto_correlation_difference(real, generated):
    """Calculates the Auto-Correlation Difference (ACD) between real and generated data."""
    def autocorrelation(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]
    
    real_acf = np.mean([autocorrelation(series[0]) for series in real], axis=0)
    generated_acf = np.mean([autocorrelation(series[0]) for series in generated], axis=0)
    
    acd = np.mean(np.abs(real_acf - generated_acf))
    return acd

def skewness_difference(real, generated):
    """Calculates the Skewness Difference (SD) between real and generated data."""
    real_skew = skew(real.reshape(-1))
    generated_skew = skew(generated.reshape(-1))
    
    sd = np.abs(real_skew - generated_skew)
    return sd

def kurtosis_difference(real, generated):
    """Calculates the Kurtosis Difference (KD) between real and generated data."""
    real_kurt = kurtosis(real.reshape(-1))
    generated_kurt = kurtosis(generated.reshape(-1))
    
    kd = np.abs(real_kurt - generated_kurt)
    return kd


if __name__ == '__main__':
    # Example usage:
    # real_data and generated_data should be numpy arrays of shape (batch_size, 1, length)
    real_data = np.random.normal(0, 1, (10, 1, 100))  # example real data
    generated_data = np.random.normal(0, 1, (10, 1, 100))  # example generated data

    print("Marginal Distribution Difference:", marginal_distribution_difference(real_data, generated_data))
    print("Auto-Correlation Difference:", auto_correlation_difference(real_data, generated_data))
    print("Skewness Difference:", skewness_difference(real_data, generated_data))
    print("Kurtosis Difference:", kurtosis_difference(real_data, generated_data))
