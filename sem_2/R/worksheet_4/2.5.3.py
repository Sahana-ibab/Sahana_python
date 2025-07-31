
import stat_fn as sf

def main():
    data = [40.0, 54.5, 34.5, 47.0, 54.0, 40.1, 34.0, 43.0, 33.0, 42.0, 44.3, 59.9, 54.0, 53.9, 62.6, 48.0, 41.8, 54.1,
            53.6, 33.0, 45.7, 56.9, 43.1, 40.6, 58.0, 52.4, 56.6, 45.0, 37.9, 59.0]

    mean_val = sf.arithmetic_mean(data)
    median_val = sf.median(data)
    mode_val = sf.mode(data)
    variance_val = sf.variance(data, mean_val)
    std_dev = sf.standard_deviation(variance_val)
    iqr = sf.inter_quartile(data)
    cv = sf.coeffi_var(std_dev,mean_val)
    data_range = sf.range_fn(data)
    skew = sf.skewness(data, mean_val, std_dev)
    kurt = sf.kurtosis(data, mean_val, std_dev)

    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Mode: {mode_val}")
    print(f"Variance: {variance_val:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Coefficient of Variation: {cv:.2f}%")
    print(f"Inter-quartile Range: {iqr:.2f}")
    print(f"Range: {data_range}")
    print(f"Skewness: {skew:.2f}")
    print(f"Kurtosis: {kurt:.2f}")

    # sf.plot_fn(data)
    sf.whisker(data)
    ct, reason = sf.central_tendency(skew)
    print(f"Suggested measure of central tendency: {ct}. Reason: {reason}")

if __name__ == '__main__':
    main()