import stat_fn as sf

def main():
    data = [
        11.23, 15.82, 29.63, 27.74, 20.42, 22.35, 10.12, 34.78, 39.91, 35.09,
        32.66, 32.60, 38.38, 37.03, 36.21, 27.00, 16.39, 44.20, 27.44, 13.09,
        17.29, 33.03, 56.20, 9.69, 28.94, 32.45, 20.11, 37.38, 25.35, 34.91,
        21.77, 27.99, 31.62, 22.36, 32.63, 22.68, 30.31, 26.52, 46.16, 46.01,
        56.61, 24.47, 29.39, 40.71, 18.52, 27.80, 19.49, 38.04, 30.88, 30.04,
        25.91, 18.54, 25.51
    ]

    mean_value = sf.arithmetic_mean(data)
    geometric_mean_value = sf.geometric_mean(data)
    harmonic_mean_value = sf.harmonic_mean(data)
    median_value = sf.median(data)
    mode_value = sf.mode(data)
    variance_value = sf.variance(data, mean_value)
    std_dev_value = sf.standard_deviation(variance_value)
    skewness_value = sf.skewness(data, mean_value, std_dev_value)
    kurt_value = sf.kurtosis(data, mean_value, std_dev_value)
    mean_dev_mean = sf.mean_deviation(data, mean_value)
    mean_dev_median = sf.mean_deviation(data, median_value)

    print(f"Mean: {mean_value}")
    print(f"Geometric Mean: {geometric_mean_value}")
    print(f"Harmonic Mean: {harmonic_mean_value}")
    print(f"Median: {median_value}")
    print(f"Mode: {mode_value}")
    print(f"Variance: {variance_value}")
    print(f"Standard Deviation: {std_dev_value}")
    print(f"Skewness: {skewness_value}")
    print(f"Kurtosis: {kurt_value}")
    print(f"Mean Deviation (Mean): {mean_dev_mean}")
    print(f"Mean Deviation (Median): {mean_dev_median}")
    num_bins = sf.num_bins(data)
    bins = sf.calculate_bins(data, num_bins)
    print(f"Frequency Distribution: {sf.frequency_distribution(data, bins)}")
    print(f"Relative Frequency Distribution: {sf.relative_frequency_distribution(data, bins)}")
    print(f"Cumulative Frequency Distribution: {sf.cumulative_frequency_distribution(data, bins)}")
    print(f"Cumulative Relative Frequency Distribution: {sf.cumulative_relative_frequency_distribution(data, bins)}")

    percentage_greater_than_40 = len([i for i in data if i > 40]) / len(data) * 100
    print(f"Percentage of measurements greater than 40: {percentage_greater_than_40:.2f}%")

    percentage_less_than_25 = len([i for i in data if i < 25]) / len(data) * 100
    print(f"Percentage of measurements less than 25: {percentage_less_than_25:.2f}%")

    print(f"Symmetry and Skewness: Based on the skewness value of {skewness_value}, the data is {'right-skewed' if skewness_value > 0 else 'left-skewed' if skewness_value < 0 else 'symmetrical'}.")
    sf.plot_frequency_polygon(data, bins)
    sf.plot_fn(data)


if __name__ == "__main__":
    main()
