import pandas as pd
ml_csv = pd.read_csv('outputs/ml-output.csv')
dl_csv = pd.read_csv('outputs/dl-output.csv')

predicted = (ml_csv['Predicted'] + dl_csv['Predicted'])/2

df = pd.DataFrame()
df["Predicted"] = predicted
df.to_csv('outputs/output.csv', index_label="Id")
