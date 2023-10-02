# Intro to Pandas

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data Import and Export](#data-import-and-export)
4. [Data Cleaning and Pre-processing](#data-cleaning-and-pre-processing)
5. [Data Exploration](#data-exploration)
6. [Data Manipulation](#data-manipulation)
7. [Data Visualization](#data-visualization)
8. [Statistical Analysis](#statistical-analysis)
9. [Linear Algebra Operations](#linear-algebra-operations)
10. [Conclusion](#conclusion)
---

Pandas is an indispensable tool for data manipulation and analysis, providing fast, flexible, and expressive data structures. It plays a vital role in cleaning, transforming, visualizing, and analyzing data, which are critical steps in making informed decisions in data science and related fields.

---

## Introduction

Pandas stands for "Python Data Analysis Library" and it is built on top of the NumPy library, which provides support for multi-dimensional arrays. The primary data structures in Pandas are Series (1-dimensional) and DataFrame (2-dimensional).

```python
import pandas as pd
import numpy as np

```

## Installation

Ensure you have Pandas installed to follow through. It's a quick installation using pip:

```bash
pip install pandas

```

## Data Import and Export

In your course, working with datasets is fundamental. Pandas provides seamless methods to read from and write to a variety of file formats.

```python
# Loading data from a CSV file
df = pd.read_csv('dataset.csv')

# Saving data to a CSV file
df.to_csv('output.csv', index=False)

```

Here, `pd.read_csv` reads a comma-separated values (csv) file into DataFrame, while `df.to_csv` writes the DataFrame to a csv file.

## Data Cleaning and Pre-processing

Before any analysis, it's crucial to clean and preprocess your data to ensure accuracy in results.

```python
# Identifying missing values
missing_values = df.isnull().sum()

# Filling missing values
df_filled = df.fillna(method='ffill')

# Converting data types
df['column_name'] = df['column_name'].astype('category')

```

- `df.isnull().sum()` returns the count of missing values for each column.
- `df.fillna(method='ffill')` fills the missing values with the preceding non-null value in the column.
- `astype('category')` converts the data type of the column to a categorical type, which often saves memory.

## Data Exploration

Understanding the structure and characteristics of your data is paramount.

```python
# Descriptive statistics
descriptive_stats = df.describe()

# Correlation matrix
correlation_matrix = df.corr()

```

- `df.describe()` provides a summary of central tendency, dispersion, and shape of the dataset’s distribution.
- `df.corr()` computes pairwise correlation of columns, excluding NA/null values, which is vital for feature selection in machine learning.

## Data Manipulation

Transforming data to get it into a suitable form for analysis or modeling is a common task.

```python
# Adding a new column
df['new_column'] = df['column1'] + df['column2']

# Filtering data
filtered_df = df[df['column'] > value]

# Grouping data
grouped_df = df.groupby('column_name').mean()

```

- `df['new_column'] = df['column1'] + df['column2']` creates a new column by adding values from two existing columns.
- `df[df['column'] > value]` filters the rows based on a condition.
- `df.groupby('column_name').mean()` groups the data by a column and computes the mean of each group, useful for aggregated analysis.

## Data Visualization

Visualizations help in understanding the distributions and relationships in data.

```python
import matplotlib.pyplot as plt

# Histogram
df['column'].hist()
plt.show()

# Scatter plot
df.plot(x='column1', y='column2', kind='scatter')
plt.show()

```

- `df['column'].hist()` plots a histogram to show the distribution of values in a column.
- `df.plot(x='column1', y='column2', kind='scatter')` plots a scatter plot to show the relationship between two columns.

## Statistical Analysis

Performing statistical analysis to derive insights from data.

```python
# Mean, Median
mean_value = df['column'].mean()
median_value = df['column'].median()

# Variance, Standard Deviation
var_value = df['column'].var()
std_value = df['column'].std()

```

- `df['column'].mean()` and `df['column'].median()` compute the mean and median of a column, respectively.
- `df['column'].var()` and `df['column'].std()` compute the variance and standard deviation, essential for understanding the data dispersion.

## Linear Algebra Operations

Linear algebra is foundational in machine learning and data science, for operations like transformations, solving systems of equations, and more.

```python
# Matrix Multiplication
result = np.dot(df1.values, df2.values)

# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(df.values)

```

- `np.dot(df1.values, df2.values)` performs matrix multiplication, crucial in various algorithms.
- `np.linalg.eig(df.values)` computes the eigenvalues and eigenvectors of a matrix, which are fundamental in

dimensionality reduction, PCA, etc.

## Conclusion

Mastering Pandas will significantly enhance your data handling and analysis capabilities, making you well-equipped to tackle the challenges posed in your course. Through a thorough understanding and application of the concepts outlined in this guide, you'll be well on your way to excelling in your data science and math endeavors.

---
