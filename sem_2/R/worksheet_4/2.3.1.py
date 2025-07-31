import stat_fn as sf

def main():

    data = [
        1.17, 1.17, 2.17, 1.00, 1.33, 0.33, 1.00, 0.83, 0.50, 1.17, 1.67, 1.17,
        0.83, 2.00, 2.17, 2.50, 2.17, 1.17, 2.17, 2.17, 2.17, 2.17, 1.17, 1.50,
        1.33, 0.33, 0.00, 1.17, 2.17, 2.17, 2.00, 2.17, 2.50, 2.17, 2.83, 1.83,
        2.17, 2.17, 2.00, 1.67, 1.50, 1.50, 1.50, 1.33, 2.00, 2.33, 1.33, 2.00,
        1.67, 1.50, 2.00, 1.33, 2.50, 2.00, 2.17, 2.17, 2.33, 2.17, 2.00, 2.17,
        1.67, 2.17, 1.50, 2.00, 2.50, 2.17, 2.17, 2.00, 2.00, 1.50, 2.33, 1.83,
        2.67, 2.33, 2.00, 1.33, 2.00, 1.50, 2.00, 2.33, 2.00, 2.50, 2.50, 2.00,
        2.00, 2.33, 2.67, 1.50, 2.00, 2.17
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
    num_bins=sf.num_bins(data)
    bins=sf.calculate_bins(data,num_bins)
    print(f"Frequency Distribution: {sf.frequency_distribution(data, bins)}")
    print(f"Relative Frequency Distribution: {sf.relative_frequency_distribution(data,bins)}")
    print(f"Cumulative Frequency Distribution: {sf.cumulative_frequency_distribution(data, bins)}")
    print(f"Cumulative Relative Frequency Distribution: {sf.cumulative_relative_frequency_distribution(data,bins)}")


    percentage_less_than_2 = len([i for i in data if i < 2.00]) / len(data) * 100
    print(f"Percentage of measurements less than 2.00: {percentage_less_than_2:.2f}%")

    proportion_greater_equal_1_5 = len([i for i in data if i >= 1.50]) / len(data)
    print(f"Proportion of subjects with measurements >= 1.50: {proportion_greater_equal_1_5:.2f}")

    percentage_between_1_50_and_1_99 = len([i for i in data if 1.50 <= i <= 1.99]) / len(data) * 100
    print(f"Percentage of measurements between 1.50 and 1.99 inclusive: {percentage_between_1_50_and_1_99:.2f}%")

    count_greater_than_2_49 = len([i for i in data if i > 2.49])
    print(f"Number of measurements greater than 2.49: {count_greater_than_2_49}")

    proportion_less_than_1_or_greater_than_2_49 = len([i for i in data if i < 1.0 or i > 2.49]) / len(data)
    print(f"Proportion of measurements less than 1.0 or greater than 2.49: {proportion_less_than_1_or_greater_than_2_49:.2f}")

    print(f"Random guess: The best guess is the mode: {mode_value}")
    sf.plot_frequency_polygon(data, bins)
    sf.plot_fn(data)

if __name__ == "__main__":
    main()
