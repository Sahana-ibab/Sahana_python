import stat_fn as sf

def main():
    data = [
        0.0269, 0.0760, 0.0990, 0.0990, 0.0990, 0.0990, 0.1050, 0.1190, 0.1460, 0.1550,
        0.1690, 0.1810, 0.2070, 0.2390, 0.2470, 0.2710, 0.2990, 0.3400, 0.3630, 0.4090,
        0.4300, 0.4680, 0.5340, 0.5930, 0.0400, 0.0850, 0.0990, 0.0990, 0.0990, 0.0990,
        0.1050, 0.1200, 0.1480, 0.1570, 0.1710, 0.1880, 0.2100, 0.2400, 0.2540, 0.2800,
        0.3000, 0.3440, 0.3660, 0.4090, 0.4360, 0.4810, 0.5340, 0.6010, 0.0550, 0.0870,
        0.0990, 0.0990, 0.0990, 0.0990, 0.1080, 0.1230, 0.1490, 0.1600, 0.1720, 0.1890,
        0.2100, 0.2420, 0.2570, 0.2800, 0.3070, 0.3480, 0.3830, 0.4100, 0.4370, 0.4870,
        0.5460, 0.6240, 0.0550, 0.0870, 0.0990, 0.0990, 0.0990, 0.0990, 0.1080, 0.1240,
        0.1490, 0.1650, 0.1740, 0.1890, 0.2140, 0.2430, 0.2600, 0.2870, 0.3100, 0.3490,
        0.3900, 0.4160, 0.4390, 0.4910, 0.5480, 0.6280, 0.0650, 0.0880, 0.0990, 0.0990,
        0.0990, 0.0990, 0.1090, 0.1340, 0.1500, 0.1650, 0.1780, 0.1920, 0.2150, 0.2450,
        0.2620, 0.2880, 0.3110, 0.3520, 0.3960, 0.4210, 0.4410, 0.4980, 0.5480, 0.6380,
        0.0670, 0.0900, 0.0990, 0.0990, 0.0990, 0.1000, 0.1090, 0.1340, 0.1500, 0.1670,
        0.1780, 0.1950, 0.2160, 0.2450, 0.2650, 0.2940, 0.3140, 0.3530, 0.3990, 0.4260,
        0.4410, 0.5030, 0.5490, 0.6600, 0.0700, 0.0720, 0.0900, 0.0990, 0.0990, 0.0990,
        0.0990, 0.0990, 0.1020, 0.1040, 0.1090, 0.1160, 0.1370, 0.1390, 0.1500, 0.1540,
        0.1670, 0.1677, 0.1790, 0.1790, 0.1970, 0.2010, 0.2260, 0.2290, 0.2460, 0.2460,
        0.2650, 0.2680, 0.2970, 0.2980, 0.3190, 0.3210, 0.3570, 0.3630, 0.4080, 0.4080,
        0.4290, 0.4290, 0.4430, 0.4540, 0.5060, 0.5220, 0.5550, 0.5920, 0.6720, 0.6820,
        0.6870, 0.7860, 0.9530, 0.6900, 0.7950, 0.9830, 0.6910, 0.8040, 0.9890, 0.6940,
        0.8200, 1.0120, 0.7040, 0.8350, 1.0260, 0.7120, 0.8770, 1.0320, 0.7200, 0.9090,
        1.0620, 0.7280, 0.9520, 1.1600
    ]

    # Calculate statistics
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

    # Print statistics
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

    poor_metabolizers = len([x for x in data if x > 0.9])
    poor_metabolizers_percentage = poor_metabolizers / len(data) * 100
    print(f"Number of poor metabolizers: {poor_metabolizers}")
    print(f"Percentage of poor metabolizers: {poor_metabolizers_percentage:.2f}%")

    less_than_0_7 = len([x for x in data if x < 0.7])
    between_0_3_and_0_6999 = len([x for x in data if 0.3 <= x <= 0.6999])
    greater_than_0_4999 = len([x for x in data if x > 0.4999])

    print(f"Number of ratios less than 0.7: {less_than_0_7}")
    print(f"Number of ratios between 0.3 and 0.6999 inclusive: {between_0_3_and_0_6999}")
    print(f"Number of ratios greater than 0.4999: {greater_than_0_4999}")

    sf.plot_frequency_polygon(data, bins)
    sf.plot_fn(data)

if __name__ == "__main__":
    main()
