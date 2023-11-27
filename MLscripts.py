import pandas as pd

df_power = pd.read_csv("/dbfs/FileStore/shared_uploads/rijaasif@microsoft.com/sample_data-1.csv")

spark_df_power = spark.createDataFrame(df_power)

spark_df_power.write.format('delta').partitionBy('deviceid').mode("overwrite").save('/mnt/delta/power_gen_data')

%sql

drop table if exists default.power_gen_data;

CREATE TABLE default.power_gen_data

USING DELTA

LOCATION '/mnt/delta/power_gen_data'

%sql 

select * from default.power_gen_data

import pandas as pd
df = spark.sql('select * from default.power_gen_data').toPandas()

model_dataset = df[['humidity','power']]
print("Here is the correlation...", model_dataset.corr())
model_dataset.plot.scatter(x = 'humidity',y='power')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

print("Now we train linear regression model based on train/test split")

eva_model = LinearRegression()
X = np.array(model_dataset['humidity']).reshape(-1, 1)
y = np.array(model_dataset['power']).reshape(-1, 1)

train_test_split_ratio = 0.7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-train_test_split_ratio)
eva_model.fit(X_train, y_train)

y_pred = eva_model.predict(X_test)
abs_error = np.mean(np.abs(y_pred - y_test))
print("Here is the absolute error", abs_error)
