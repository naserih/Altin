import pandas as pd
from lightweight_charts import Chart

file_name = './data/data_20210924_20230922.csv'
df = pd.read_csv(file_name)


if __name__ == '__main__':
    
    chart = Chart()
    
    # Columns: time | open | high | low | close | volume
    df = pd.read_csv(file_name)
    print(df.columns)
    chart.set(df)
    chart.show(block=True)