
import stat_fn as sf

def main():
    data = [500,570,560,570,450,560,570]

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
    print(f"Inter-quartile Range: {iqr}")
    print(f"Range: {data_range}")
    print(f"Skewness: {skew:.2f}")
    print(f"Kurtosis: {kurt:.2f}")

    # sf.plot_fn(data)
    sf.whisker(data)
    ct, reason = sf.central_tendency(skew)
    print(f"Suggested measure of central tendency: {ct}. Reason: {reason}")

if __name__ == '__main__':
    main()