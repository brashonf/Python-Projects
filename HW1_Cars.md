```python
"""
Created on Tue Oct 24 06:17:29 2023

@author: brashonford
"""
```

```python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
from matplotlib import pyplot as plt
```

```python
cars_df = pd.read_csv('/Users/brashonford/ANA620/HW1_Cars.csv')
```

```python
cars_df.columns
```

```python
cars_df.describe()
```

# 1. Scatter Plot Matrix: MSRP, Invoice, Horsepower, MPG Highway & MPG City


```python
import pandas
cars_df = pd.read_csv('/Users/brashonford/ANA620/HW1_Cars.csv')
cars_df.head()
```

```python
from pandas.plotting import scatter_matrix
```

```python
sns.set_theme(style='darkgrid')
sns.set_palette('dark')
```

```python
contvar=['MSRP' , 'Invoice', 'Horsepower', 'MPG_City', 'MPG_Highway']
```

```python
sns.pairplot(cars_df[contvar])
```

# Question 2: t-tests

```python
cars_df.head()
```

```python
cars_df.DriveTrain
```

```python
front_mileage = cars_df[cars_df['DriveTrain']== 'Front']['MPG_City']
awd_mileage = cars_df[cars_df['DriveTrain']== 'All']['MPG_City']
```

```python
cars_df.groupby('DriveTrain').describe()
```

```python
alpha = 0.05
```

```python
stats.ttest_ind(front_mileage , awd_mileage , equal_var=False)
```

```python
if p_value < alpha:
    print("Reject Null Hypothesis")
    print("There is a significant difference in mileage")
else:
    print("Fail to rejext null")
    print("there is no significant difference")
```

```python
from statsmodels.stats.weightstats import ttest_ind
```

Engine Size between USA and Asia vehicles

```python
engine_usa = cars_df[cars_df['Origin']=='USA']['EngineSize']
engine_asia = cars_df[cars_df['Origin']== 'Asia']['EngineSize']
alpha = 0.05
```

```python
stats.ttest_ind(engine_usa, engine_asia , equal_var=False)
```

```python
if p_value_origin < alpha:
    print("\nReject the null hypothesis for Origin.")
    print("There is a statistically significant difference in engine size between Asia and USA.")
else:
    print("\nFail to reject the null hypothesis for Origin.")
    print("There is no statistically significant difference in engine size between Asia and USA.")
```

```python
cars_df['Invoice'] = pd.to_numeric(cars_df['Invoice'], errors='coerce')
cars_df['MSRP'] = pd.to_numeric(cars_df['MSRP'], errors='coerce')
```

```python
cost_sedan = cars_df[cars_df['Type']== 'Sedan']['Invoice']
cost_wagon = cars_df[cars_df['Type']== 'Wagon']['Invoice']
alpha = 0.05
```

```python
stats.ttest_ind(cost_sedan , cost_wagon, equal_var=False)
```

```python
if p_value_invoice < alpha:
    print("\nReject the null hypothesis for Invoice.")
    print("There is a statistically significant difference in Invoice cost between Wagon and Sedan vehicles.")
else:
    print("\nFail to reject the null hypothesis for Invoice.")
    print("There is no statistically significant difference in Invoice cost between Wagon and Sedan vehicles.")
```

```python
cost_usa = cars_df[cars_df['Origin']== 'USA']['MSRP']
cost_europe = cars_df[cars_df['Origin']== 'Europe']['MSRP']
alpha = 0.05
```

```python
t_stat_origin , p_value_origin = stats.ttest_ind(cost_usa , cost_europe , equal_var=False)
```

```python
if p_value_origin < alpha:
    print("\nReject the null hypothesis for origin.")
    print("There is a statistically significant difference in MSRP between USA and Europe.")
else:
    print("\nFail to reject the null hypothesis for origin.")
    print("There is no statistically significant difference in MSRP between USA and Europe.")
```

# Question 3: Estimates & Standard Error

```python
from scipy.stats import sem 
import statistics
```

* MPG_City

```python
data1 = cars_df.MPG_City
```

```python
sem(data1)
```

```python
len(cars_df.MPG_City)
```

```python
cars_df.MPG_City.mean()
```

```python
confidence_interval=0.95
margin_error = 0.2531 * stats.t.ppf((1 + 0.95)/ 2, 428-1)
print(margin_error)
```

```python
lower_bound = 20.060 - margin_error
upper_bound = 20.060 + margin_error
print(lower_bound, upper_bound)
```

* Mean Horsepower

```python
cars_df.Horsepower.mean()
```

```python
len(cars_df.Horsepower)
```

```python
sem(cars_df.Horsepower)
```

```python
confidence_interval=0.95
margin_error = 3.472 * stats.t.ppf((1 + 0.95)/ 2, 428-1)
print(margin_error)
```

```python
lower_bound = 215.88 - margin_error
upper_bound = 215.88 + margin_error
print(lower_bound, upper_bound)
```

```python
from statistics import variance
```

```python
sample1 = cars_df.Horsepower
print(variance(sample1))
```

* MPG_Highway Variance in Asia

```python
cars_df.head()
```

```python
asian_cars = cars_df[cars_df['Origin']== 'Asia']
```

```python
asian_cars['MPG_Highway'].var()
```

```python
cars_df['MSRP']=cars_df['MSRP'].str.replace('[$,]', '', regex=True).astype(float)
```

```python
cars_usa = cars_df[cars_df['Origin']== 'USA']
```

```python
msrp_var = cars_usa['MSRP'].var()
print(msrp_var)
```

# Question 4: Regression to predict gas milage 


* Simple Model using one continuous variable

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

```python
cars_df.head()
```

```python
X = cars_df[['Horsepower']]
X = sm.add_constant(X)
y = cars_df['MPG_Highway']
```

```python
model2 = sm.OLS(y,X).fit()
```

```python
print(model2.summary())
```

```python
sns.pairplot(cars_df[['Horsepower', 'MPG_Highway']])
```

```python
cars_df.describe()
```

Multiple Regression with 3 variables

```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
```

```python
X = cars_df[['Length' , 'Weight' , 'Horsepower']]
X = sm.add_constant(X)
y = cars_df['MPG_City']
```

```python
model = sm.OLS(y,X).fit()
```

```python
print(model.summary())
```

```python
import seaborn as sns
sns.pairplot(cars_df[['Horsepower', 'Weight', 'Length']])
```

```python

```
