#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
infodata = pd.read_csv('C:/Users/moham/Untitled Folder/Life-Expectancy-Data.csv')





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Life-Expectancy-Data.csv')

# Remove any leading or trailing spaces in the column names
df.columns = df.columns.str.strip()

# Check if the 'Life expectancy' column is present in the dataframe
if 'Life expectancy' not in df.columns:
    print("Error: 'Life expectancy' column not found in dataset.")
else:
    # Visualize the distribution of the target variable
    sns.histplot(df['Life expectancy'])

    # Preprocess the data by dropping missing values and scaling the features
    df.dropna(inplace=True)
    features = df.drop(['Country', 'Year', 'Status', 'Life expectancy'], axis=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Convert the scaled features back to a Pandas dataframe
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Visualize the correlation between the features and the target variable
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')








import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
df = pd.read_csv('Life-Expectancy-Data.csv')

# Remove any leading or trailing spaces in the column names
df.columns = df.columns.str.strip()

# Preprocess the data by dropping missing values and scaling the features
df.dropna(inplace=True)
features = df.drop(['Country', 'Year', 'Status', 'Life expectancy'], axis=1)

# Create dummy variables for the categorical feature 'Status'
df = pd.get_dummies(df, columns=['Status'])

# Split the data into training and testing sets
target = df['Life expectancy']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Define a function to display the selected plot
def plot_func(plot_type):
    if plot_type == 'Distribution of Life Expectancy':
        plt.hist(target, bins=20)
        plt.title('Distribution of Life Expectancy')
        plt.xlabel('Life Expectancy')
        plt.ylabel('Frequency')
        plt.show()

    elif plot_type == 'Life Expectancy vs. GDP per Capita':
        plt.scatter(df['GDP'], df['Life expectancy'], c=df['Status_Developed'], cmap='viridis', alpha=0.5)
        plt.xscale('log')
        plt.title('Life Expectancy vs. GDP per Capita')
        plt.xlabel('GDP per Capita')
        plt.ylabel('Life Expectancy')
        plt.colorbar()
        plt.show()

    elif plot_type == 'Top 10 Countries by Life Expectancy':
        top_10 = df.sort_values(by='Life expectancy', ascending=False).head(10)
        plt.bar(top_10['Country'], top_10['Life expectancy'])
        plt.title('Top 10 Countries by Life Expectancy')
        plt.xlabel('Country')
        plt.ylabel('Life Expectancy')
        plt.xticks(rotation=90)
        plt.show()

    elif plot_type == 'Life Expectancy vs. Alcohol Consumption':
        plt.scatter(df['Alcohol'], df['Life expectancy'], color='blue', alpha=0.5)
        plt.title('Life Expectancy vs. Alcohol Consumption')
        plt.xlabel('Alcohol Consumption')
        plt.ylabel('Life Expectancy')
        plt.show()

# Create a dropdown widget for selecting the plot
plot_type_dropdown = widgets.Dropdown(options=['Distribution of Life Expectancy', 'Life Expectancy vs. GDP per Capita', 'Top 10 Countries by Life Expectancy', 'Life Expectancy vs. Alcohol Consumption'], value='Distribution of Life Expectancy', description='Select a plot:')

# Attach the function to the dropdown
widgets.interactive(plot_func, plot_type=plot_type_dropdown)


# In[1]:


import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

# Load the dataset
df = pd.read_csv('Life-Expectancy-Data.csv')

# Remove any leading or trailing spaces in the column names
df.columns = df.columns.str.strip()

# Preprocess the data by dropping missing values and scaling the features
df.dropna(inplace=True)
features = df.drop(['Country', 'Year', 'Status', 'Life expectancy'], axis=1)

# Create dummy variables for the categorical feature 'Status'
df = pd.get_dummies(df, columns=['Status'])

# Define a function to display the selected plot
def plot_func(plot_type):
    if plot_type == 'Distribution of Life Expectancy':
        fig = px.histogram(df, x='Life expectancy', nbins=20)
        fig.update_layout(title='Distribution of Life Expectancy', xaxis_title='Life Expectancy', yaxis_title='Frequency')
        fig.show()

    elif plot_type == 'Life Expectancy vs. GDP per Capita':
        fig = px.scatter(df, x='GDP', y='Life expectancy', color='Status_Developed', log_x=True, opacity=0.5)
        fig.update_layout(title='Life Expectancy vs. GDP per Capita', xaxis_title='GDP per Capita', yaxis_title='Life Expectancy', coloraxis_colorbar_title='Status')
        fig.show()

    elif plot_type == 'Top 10 Countries by Life Expectancy':
        top_10 = df.sort_values(by='Life expectancy', ascending=False).head(10)
        fig = px.bar(top_10, x='Country', y='Life expectancy')
        fig.update_layout(title='Top 10 Countries by Life Expectancy', xaxis_title='Country', yaxis_title='Life Expectancy')
        fig.show()

    elif plot_type == 'Life Expectancy vs. Alcohol Consumption':
        fig = px.scatter(df, x='Alcohol', y='Life expectancy', opacity=0.5)
        fig.update_layout(title='Life Expectancy vs. Alcohol Consumption', xaxis_title='Alcohol Consumption', yaxis_title='Life Expectancy')
        fig.show()

# Create a dropdown widget for selecting the plot
plot_type_dropdown = widgets.Dropdown(options=['Distribution of Life Expectancy', 'Life Expectancy vs. GDP per Capita', 'Top 10 Countries by Life Expectancy', 'Life Expectancy vs. Alcohol Consumption'], value='Distribution of Life Expectancy', description='Select a plot:')

# Attach the function to the dropdown
widgets.interactive(plot_func, plot_type=plot_type_dropdown)











