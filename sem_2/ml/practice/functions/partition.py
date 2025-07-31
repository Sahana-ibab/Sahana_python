import pandas as pd

# for loading dataset:
def data_load():
    df = pd.read_csv('/home/ibab/simulated_data_multiple_linear_regression_for_ML.csv')

    # X= df[["age"]]
    # y= df["disease_score"]
    return df

# function for partition :
def partition_data(df, feature, threshold):

    lower_data=df[df[feature]<=threshold]
    upper_data=df[df[feature]>threshold]
    return lower_data, upper_data

def main():
    df= data_load()

    lower_80, upper_80 = partition_data(df,"BP", 80)
    lower_80.to_csv("/home/ibab/simulated_data_partitions/lower_80.csv", index=False)
    upper_80.to_csv("/home/ibab/simulated_data_partitions/upper_80.csv", index=False)
    print("Task completed!")

    # test cases:
    lower_78, upper_78 = partition_data(df, "BP", 78)
    lower_82, upper_82 = partition_data(df, "BP", 82)

    lower_78.to_csv("/home/ibab/simulated_data_partitions/lower_78.csv", index=False)
    upper_78.to_csv("/home/ibab/simulated_data_partitions/upper_78.csv", index=False)
    lower_82.to_csv("/home/ibab/simulated_data_partitions/lower_82.csv", index=False)
    upper_82.to_csv("/home/ibab/simulated_data_partitions/upper_82.csv", index=False)

if __name__ == '__main__':
    main()




