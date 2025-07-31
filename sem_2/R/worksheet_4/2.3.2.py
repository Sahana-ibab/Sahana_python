import stat_fn as sf

def main():
    data = [
        8.1, 20.0, 10.0, 4.0, 5.0, 4.0, 7.0, 6.0, 10.0, 3.0, 6.0, 7.0, 10.0, 6.0, 6.0,
        4.0, 21.0, 5.0, 10.0, 7.0, 15.0, 3.0, 3.0, 16.0, 9.0, 3.0, 15.0, 3.8, 30.0, 22.0,
        7.5, 8.0, 6.0, 15.0, 6.0, 5.0, 4.0, 4.0, 6.0, 25.0, 15.0, 8.3, 7.5, 3.0, 5.0, 5.0,
        6.0, 3.0, 7.0, 5.0, 4.0, 2.0, 10.0, 2.0, 5.0, 5.0, 5.0, 5.0, 18.0, 8.0, 16.0, 16.0,
        3.0, 6.0, 10.0, 5.0, 7.0, 7.0, 15.0, 10.0, 5.0, 6.0, 7.0, 8.0, 5.0, 5.0, 8.0, 25.0,
        10.0, 5.0, 8.0, 9.0, 5.0, 10.0, 12.0, 8.0, 3.0, 5.0, 5.0, 14.0, 5.0, 8.0, 10.0, 6.0,
        3.0, 3.0, 10.0, 5.0, 5.0, 4.0, 8.0, 7.0, 8.0, 5.0, 3.0, 3.0, 11.0, 12.0, 9.0, 10.0,
        5.0, 7.0, 6.0, 4.0, 6.0, 8.0, 5.0, 15.0, 10.0, 7.0, 7.0, 15.0, 4.0, 10.0, 8.0, 30.0,
        12.0, 9.0, 5.0, 4.0, 5.0, 2.0, 9.0, 5.0, 5.0, 8.0, 13.0, 6.0, 5.0, 6.0, 19.0, 8.0,
        8.0, 8.0, 3.3, 8.0, 6.0, 9.2, 10.0, 4.0, 14.0, 4.0, 14.0, 10.0, 8.0, 7.0, 10.0, 8.0,
        6.0
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
    percentage_between_10_and_14_9 = len([i for i in data if 10.0 <= i <= 14.9]) / len(data) * 100
    print(f"Percentage of measurements between 10.0 and 14.9 inclusive: {percentage_between_10_and_14_9:.2f}%")

    count_less_than_20 = len([i for i in data if i < 20])
    print(f"Number of observations less than 20: {count_less_than_20}")

    proportion_greater_than_or_equal_25 = len([i for i in data if i >= 25]) / len(data)
    print(f"Proportion of measurements greater than or equal to 25: {proportion_greater_than_or_equal_25:.2f}")

    percentage_less_than_10_or_greater_than_19_95 = len([i for i in data if i < 10.0 or i > 19.95]) / len(data) * 100
    print(f"Percentage of measurements less than 10.0 or greater than 19.95: {percentage_less_than_10_or_greater_than_19_95:.2f}%")

    print(f"Distribution of aneurysm sizes: Mean = {mean_value}, Median = {median_value}, Mode = {mode_value}, Skewness = {skewness_value}, Kurtosis = {kurt_value}")

    sf.plot_frequency_polygon(data, bins)
    sf.plot_fn(data)


if __name__ == "__main__":
    main()
