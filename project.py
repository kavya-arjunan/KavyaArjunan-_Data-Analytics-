#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv('mobile_recommendation_system_dataset.csv')


# # EDA

# In[5]:


df


# In[6]:


df.head(5)


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.info()


# In[10]:


df.columns


# In[11]:


df.dtypes


# # null values

# In[12]:


cols=df.columns


# In[13]:


sns.heatmap(df[cols].isnull())


# In[14]:


df.isnull().sum()


# In[15]:


df=df.dropna()


# # finding duplicate rows

# In[16]:


#finding duplicate rows
duplicate_rows=df[df.duplicated(keep='first')]

#num of duplicate rows
num_duplicate = duplicate_rows.shape[0]

print(f'Number Of Duplicate Rows:{num_duplicate}')
duplicate_rows


# # Remove the duplicate rows and update the DataFrame

# In[17]:


df.drop_duplicates(keep='first',inplace=True)


# # corpus

# In[18]:


#extract the information for 'storage' and 'RAM' using regular expressions
extracted_data = df['corpus'].str.extract(r'Storage(\d+)\s+GBRAM(\d+)',expand=True)

#convert the extracted values to numeric values
df['Storage'] = pd.to_numeric(extracted_data[0],errors='coerce')
df['RAM'] = pd.to_numeric(extracted_data[1],errors='coerce')

#now the 'storage' and 'RAM' columns created with the extracted values
print(df[['Storage','RAM']])


# In[19]:


del df["corpus"]


# # Price

# In[20]:


df["price"] = df["price"].replace('₹', '', regex=True)
df["price"]


# In[21]:


df["price"] = df["price"].str.replace(',','')


# In[22]:


df["price"] = df["price"].astype(int)


# In[23]:


df["price"]


# # Name

# In[24]:


df["name"] = df["name"].str.replace(r'\s*\(.+\)', '', regex=True)
print(df["name"])


# In[25]:


df


# In[26]:


df.dropna()


# # Visualization

# In[27]:


sns.pairplot(df)
plt.show()


# In[28]:


fig, ax =plt.subplots()
sns.boxplot(y='price', data=df)
plt.show()


# In[29]:


sns.histplot(df['price'],bins=20)
plt.xlabel('price')
plt.ylabel('frequency')
plt.show()


# In[30]:


fig, ax =plt.subplots()
sns.boxplot(y='RAM', data=df)
plt.show()


# # Outliers

# In[31]:


df = df[df["price"] < 50000]


# In[32]:


df = df[df["RAM"] < 60]


# In[33]:


# Distribution of Popularity
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], kde=True)
plt.title('Distribution of price')
plt.xlabel('price')
plt.ylabel('Count')
plt.show()


# In[34]:


sns.relplot(x = "price" , y = "Storage" , data = df , kind = "scatter" , hue = "ratings" ) # scatter
plt.show()


# In[35]:


sns.relplot(x = "price" , y = "RAM" , data = df , kind = "scatter" , hue = "ratings" ) # scatter
plt.show()


# In[36]:


plt.figure(figsize=(30,20))
ax = sns.histplot(x="price" , data=df, bins=30 ,  palette ='crest_r')
plt.show()


# In[37]:


sns.catplot(x = "price"  , y = "ratings", data = df , hue = "RAM" ,kind = "point" )


# In[38]:


df


# # Accuracy

# In[39]:


del df["name"]


# In[40]:


del df["imgURL"]


# In[41]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# For example, convert categorical variables to numerical form, handle missing values, etc.

# Separate the features (X) and the target variable (y)
X = df.drop(columns=["price"])
y = df["price"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a regression model (e.g., Linear Regression)
reg_model = LinearRegression()

# Train the model on the training set
reg_model.fit(X_train, y_train)

# Predict prices on the testing set
y_pred = reg_model.predict(X_test)

# Evaluate the model's accuracy using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Optionally, you can also calculate the R-squared value to assess the model's goodness of fit
r_squared = reg_model.score(X_test, y_test)
print("R-squared:", r_squared)


# In[42]:


submit = pd.DataFrame()
submit['Actual_price']=y_test
submit['predict_price']=y_pred


# In[43]:


submit.head()


# In[44]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# For example, convert categorical variables to numerical form, handle missing values, etc.

# Separate the features (X) and the target variable (y)
X = df.drop(columns=["price"])
y = df["price"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Random Forest Regression model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training set
rf_model.fit(X_train, y_train)

# Predict prices on the testing set
y_pred = rf_model.predict(X_test)

# Evaluate the model's accuracy using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[45]:


new=pd.DataFrame()
new['Actual_price']=y_test
new['predict_price']=y_pred

new.head()


# In[46]:


get_ipython().system('pip install keras')


# In[47]:


get_ipython().system('pip install tensorflow')


# In[48]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error

# For example, convert categorical variables to numerical form, handle missing values, etc.

# Separate the features (X) and the target variable (y)
X = df[["Storage","RAM"]].values
y = df["price"]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Create the Neural Network model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Output layer with 1 neuron for regression

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model on the training set
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

# Predict prices on the testing set
y_pred = model.predict(X_test_scaled).flatten()

# Evaluate the model's accuracy using Mean Squared
# Evaluate the model's accuracy using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[ ]:




