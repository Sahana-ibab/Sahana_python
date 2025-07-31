import pandas as pd
def data_normalization(data):
    min=data.min()
    max=data.max()

    norm_data=((data-min)/(max-min))
    return norm_data



