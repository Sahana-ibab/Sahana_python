import stat_fn as sf

def main():
    data = [
        33.57, 38.34, 26.86, 27.78, 40.81, 29.01, 47.78, 54.33, 28.99, 25.21, 36.42, 24.54,
        24.49, 29.07, 26.54, 31.44, 30.49, 41.50, 41.75, 33.23, 28.21, 27.74, 30.08, 33,
        27.38, 29.39, 44.68, 47.09, 42.10, 33.48
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

    percentage_less_than_30 = len([i for i in data if i < 30]) / len(data) * 100
    print(f"Percentage of measurements less than 30: {percentage_less_than_30:.2f}%")

    percentage_between_40_and_49_99 = len([i for i in data if 40.0 <= i <= 49.99]) / len(data) * 100
    print(f"Percentage of measurements between 40.0 and 49.99 inclusive: {percentage_between_40_and_49_99:.2f}%")

    percentage_greater_than_34_99 = len([i for i in data if i > 34.99]) / len(data) * 100
    print(f"Percentage of measurements greater than 34.99: {percentage_greater_than_34_99:.2f}%")

    print(
        f"Symmetry and Skewness: Based on the skewness value of {skewness_value}, the data is {'right-skewed' if skewness_value > 0 else 'left-skewed' if skewness_value < 0 else 'symmetrical'}.")

    count_less_than_40 = len([i for i in data if i < 40])
    print(f"Number of measurements less than 40: {count_less_than_40}")
    sf.plot_frequency_polygon(data, bins)
    sf.plot_fn(data)


if __name__ == "__main__":
    main()
