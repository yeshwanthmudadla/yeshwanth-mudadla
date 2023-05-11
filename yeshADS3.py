

#importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import curve_fit


# load the dataset
df = pd.read_csv(r"C:\Users\yeshwanth\OneDrive\Desktop\ADS3\API_AG.LND.FRST.ZS_DS2_en_csv_v2_5358376.csv", header=2)

# Functions From Practical class

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1960, 2010)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True) # replace NaN values with 0

# Find the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)



# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['2000'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('Forest area Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('Forest area (% of land area) in 2000')

# Add legend
plt.legend()

# Show the plot
plt.show()


# Create a list of countries in each cluster
country_clusters = []
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    cluster_countries['Cluster'] = f'Cluster {i}'
    country_clusters.append(cluster_countries)

# Concatenate all the dataframes
all_countries = pd.concat(country_clusters)

# Group the dataframe by cluster
grouped = all_countries.groupby('Cluster')
    
    # Concatenate the dataframes in country_clusters
all_countries = pd.concat(country_clusters)

# Group the dataframe by cluster
grouped = all_countries.groupby('Cluster')

# Display the result
for i, group in grouped:
    print(f'Countries in Cluster {i}:')
    display(group)
    print()


def linear_model(x, a, b):
    return a*x + b


# Define the columns to use
columns_to_use = [str(year) for year in range(1960, 2000)]


# Select a country
country = 'China'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values[0]
x_data = np.array(range(1960, 2000))
y_data = country_data

# Remove any NaN or inf values from y_data
y_data = np.nan_to_num(y_data)

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)


def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper


# Predicting future values and corresponding confidence intervals
x_future = np.array(range(1960, 2050))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, alpha=0.3, label='Confidence Interval')
plt.xlabel('Year')
plt.ylabel('Forest area (% of land area)')
plt.title(f'{country} Forest Land Fitting')
plt.legend()
plt.show()



#pie chart comparing agriculture land in Denmark to other land
# Select the country and year of interest
country = 'China'
year = '2016'

# Filter the dataframe by country and year
df_country = df[df['Country Name'] == country]
value = df_country[year].values[0]

# Plot the pie chart
labels = [f'Forest land ({year})', 'Other land']
sizes = [value, 100 - value]
explode = (0.1, 0)  # explode the first slice
colors = ['#8ea594','#abad71']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=45)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title(f'{country}: {value}% of land is forest ({year})')
plt.show()

