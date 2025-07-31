import math as mt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def standard_deviation(V):
    SD=mt.sqrt(V)
    return SD

def variance(x,m):
    V=0
    for i in x:
        V+=(i-m)**2
    V = V/(len(x)-1)
    return V

def mean_deviation(x, M):
    MD = 0
    for i in x:
        MD += abs(i-M)
    # print(MD)
    MD = MD / len(x)
    return MD

def mode(x):
    f = {}
    for i in x:
        f[i] = f.get(i, 0) + 1

    max_count = max(f.values())
    modes = [key for key, value in f.items() if value == max_count]

    if max_count == 1:
        return "No mode"

    return modes if len(modes) > 1 else modes[0]

def plot_fn(x):
    def bin_no(x):
        B= int(1 + np.log2(len(x)))
        if B > len(x) // 2:
            B = len(x) // 2
        return B
    plt.hist(x,bins=bin_no(x), edgecolor="black")
    plt.show()

def whisker(x):
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=x, color='skyblue')
    plt.title("Box-and-Whisker Plot Example", fontsize=14)
    plt.ylabel("Values", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def median(x):
    x=sorted(x)
    if len(x)%2==1:
        M=x[((len(x)+1)//2)-1]
    else:
        M=(x[(len(x)//2)-1]+x[(len(x)//2+1)-1])/2
    return M

def geometric_mean(x):
    GM=1
    for i in x:
        GM*=i
    GM=GM**(1/len(x))
    return GM

def harmonic_mean(x):
    if 0 in x:
        return "Not Defined: data contains 0!"
    HM = 0
    for i in x:
        HM += 1/i
    # print(HM)
    HM = len(x)/HM
    return HM

def arithmetic_mean(x):
    AM=0
    for i in x:
        AM+=i
    # print(AM)
    AM=AM/len(x)
    return AM


def skewness(x,m,SD):
    n = len(x)
    skew = sum((i - m) ** 3 for i in x) / ((n - 1) * (SD ** 3))
    return skew


def kurtosis(x,m, SD):
    n = len(x)
    kurt = sum((i - m) ** 4 for i in x) / ((n - 1) * (SD ** 4))
    return kurt -3

def inter_quartile(x):
    sorted_x = sorted(x)
    n = len(sorted_x)
    q1 = median(sorted_x[:n // 2])
    q3 = median(sorted_x[(n + 1) // 2:])
    return q3 - q1

def coeffi_var(SD, m):
    cov = (SD / m) * 100
    return cov

def range_fn(x):
    return max(x) - min(x)

def num_bins(x):
    B= int(1 + np.log2(len(x)))
    if B > len(x) // 2:
        B = len(x) // 2
    return B

def calculate_bins(data, num_bins):
    data_range = max(data) - min(data)
    bin_width = data_range / num_bins
    bins = np.arange(min(data), max(data) + bin_width, bin_width)  # Create bins
    return bins

def frequency_distribution(data, bins):
    return pd.cut(data, bins=bins).value_counts().sort_index()

def relative_frequency_distribution(data, bins):
    return pd.cut(data, bins=bins).value_counts().sort_index()

def cumulative_frequency_distribution(data, bins):
    return pd.cut(data, bins=bins).value_counts().cumsum().sort_index()

def cumulative_relative_frequency_distribution(data, bins):
    return pd.cut(data, bins=bins).value_counts().cumsum().sort_index()

def plot_frequency_polygon(data, bins):
    freq_dist = frequency_distribution(data, bins)
    midpoints = [interval.mid for interval in pd.cut(data, bins).categories]
    plt.plot(midpoints, freq_dist.values, marker='o', linestyle='-', color='b')
    plt.title('Frequency Polygon')
    plt.xlabel('S/R Ratio')
    plt.ylabel('Frequency')
    plt.show()

def central_tendency(skew):
    if skew < -1:
        ct = "Median"
        reason = "Data is heavily skewed to the left."
    elif skew > 1:
        ct = "Median"
        reason = "Data is heavily skewed to the right."
    else:
        ct = "Mean"
        reason = "Data is approximately symmetric."
    return ct, reason

def main():
    # x=[6,1,2,3,4,5]
    x = [44, 46, 49, 52, 55, 62, 67, 72, 77, 80, 83, 86, 88, 90, 92, 94, 99, 100, 101, 106]
    y = [44, 46, 1, 5, 44, 62, 67, 72, 177, 80, 56, 86, 88, 90, 92, 94, 99, 106, 10, 106]

    m=arithmetic_mean(x)
    print("Arithmetic mean: ", m)
    print("Harmonic_mean:",harmonic_mean(x))
    print("geometric_mean(x): ",geometric_mean(x))
    M=median(x)
    print("Median: ", M)
    print("Mean deviation: ",mean_deviation(x, M))
    V=variance(x, m)
    print("Mode: ", mode(x))
    print("Variance: ", V)
    SD = standard_deviation(V)
    print("Standard deviation: ", SD)
    m2=arithmetic_mean(y)
    print("Coefficient of variance: ",coeffi_var(SD, m))
    print("Inter- Quartiles: ",inter_quartile(x))
    print("skewness: ",skewness(x, m, SD))
    print("Kurtosis: ",kurtosis(x, m, SD))
    plot_fn(x)
    whisker(x)


if __name__ == '__main__':
    main()