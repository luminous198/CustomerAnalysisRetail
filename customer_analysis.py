import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from sklearn.linear_model import LinearRegression
import os


IMAGE_DIRNAME = r'C:\Users\milan\Documents\projects\retail-customer-analysis\plots'

def plot_transaction_type(df):
    counts = df['Transaction'].value_counts()
    sns.barplot(y=counts.values, x=counts.index, width=0.5)
    plt.xlabel('Description')
    plt.ylabel('Count')
    plt.title('Transaction Types')
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'transaction_type.png'))
    plt.show()

def plot_time_of_day_bar(df):
    sns.countplot(data=df, x='Time of Day', palette="pastel")
    # Create a circle for the center of the pie chart
    circle = plt.Circle((0, 0), 0.7, color='white')
    # Add the circle to the plot
    plt.gca().add_artist(circle)
    # Add a title
    plt.title("Time of Day")
    # Display the plot
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'time_of_day.png'))
    plt.show()

def top_invoice_customer_timings(df):
    invoice_counts = df['InvoiceNo'].value_counts()
    max_invoice_no = invoice_counts.idxmax()
    max_invoice_customer = df[df['InvoiceNo'] == max_invoice_no]
    max_customer_id = max_invoice_customer['CustomerID'].values[0]
    print("Customer ID with Maximum Transactions:", max_customer_id)

    counts = max_invoice_customer['Time of Day'].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title('Time of Day Distribution for Customer ID with Maximum Transactions ')
    plt.legend(title='Time of Day', loc='best', bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'timings_of_top_customer.png'))
    plt.show()

def create_description_wordcloud(df):

    text = ' '.join(df['Description'].dropna().astype(str).values)
    wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title('Word Cloud of Description Column')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'desc_wordcloud.png'))
    plt.show()

def top_product_descriptions(df):
    mean_prices = df.groupby('Description')['UnitPrice'].mean().reset_index()
    top_descriptions = df['Description'].value_counts().head(10).reset_index()
    top_descriptions.columns = ['Description', 'Count']
    top_descriptions = top_descriptions.merge(mean_prices, on='Description')
    plt.figure(figsize=(14, 8))
    barplot = sns.barplot(y='Description', x='Count', data=top_descriptions)
    wrapped_labels = [f"{row['Description']} ({row['UnitPrice']:.2f})" for _, row in top_descriptions.iterrows()]
    for index, label in enumerate(wrapped_labels):
        barplot.text(top_descriptions['Count'][index] / 2, index, label, color='white', ha="center", va="center")
    barplot.set_yticklabels([])
    plt.subplots_adjust(left=0.2)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'top_products_v2.png'))
    plt.show()

def make_trending_products(df2):
    df = df2.copy()
    # find month over month growth
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')

    df = df[df['Quantity'] > 0]

    monthly_sales = df.groupby(['MonthYear', 'Description'])['Quantity'].sum().reset_index()
    monthly_sales['MonthYear'] = monthly_sales['MonthYear'].dt.to_timestamp()
    monthly_sales = monthly_sales.sort_values(by=['Description', 'MonthYear'])
    monthly_sales['PrevMonthSales'] = monthly_sales.groupby('Description')['Quantity'].shift(1)
    monthly_sales['SalesGrowth'] = monthly_sales['Quantity'] - monthly_sales['PrevMonthSales']
    monthly_sales['SalesGrowth'] = monthly_sales['SalesGrowth'].fillna(0)

    # find trending products Define a threshold for significant increase and decline
    increase_threshold = 5
    decline_threshold = -5
    trending_products = []
    descriptions = monthly_sales['Description'].unique()
    for desc in descriptions:
        product_sales = monthly_sales[monthly_sales['Description'] == desc]
        increasing = False

        for i in range(1, len(product_sales)):
            if product_sales.iloc[i]['SalesGrowth'] > increase_threshold:
                increasing = True
            if increasing and product_sales.iloc[i]['SalesGrowth'] < decline_threshold:
                trending_products.append(desc)
                break

    # Calculate percentage sales growth
    monthly_sales['PercentageSalesGrowth'] = (monthly_sales['SalesGrowth'] / monthly_sales['PrevMonthSales']) * 100
    monthly_sales['PercentageSalesGrowth'].replace([float('inf'), -float('inf')], 0, inplace=True)
    monthly_sales['PercentageSalesGrowth'].fillna(0, inplace=True)

    # find top 10 products with trends
    trending_sales = monthly_sales[monthly_sales['Description'].isin(trending_products)]
    trending_sales['AbsoluteSalesGrowth'] = trending_sales['SalesGrowth'].abs()
    product_growth = trending_sales.groupby('Description')['AbsoluteSalesGrowth'].sum().reset_index()
    top_10_products = product_growth.sort_values(by='AbsoluteSalesGrowth', ascending=False).head(9)
    top_10_trending_sales = trending_sales[trending_sales['Description'].isin(top_10_products['Description'])]

    # Create a FacetGrid to plot multiple charts in a grid
    g = sns.FacetGrid(top_10_trending_sales, col='Description', col_wrap=3, height=4, aspect=1.5,
                      sharey=False)
    g.map(sns.lineplot, 'MonthYear', 'PercentageSalesGrowth')
    g.set_titles('{col_name}')
    g.set_axis_labels('Month-Year', 'Sales Growth Percentage')
    g.set_xticklabels(rotation=45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Sales Trends of Top Trending Products Based on Percentage Growth', fontsize=16)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'trending_products_percent_growth.png'))
    plt.show()

    # Create a FacetGrid to plot multiple charts in a grid
    g = sns.FacetGrid(top_10_trending_sales, col='Description', col_wrap=3, height=4, aspect=1.5,
                      sharey=False)
    g.map(sns.lineplot, 'MonthYear', 'Quantity')
    g.set_titles('{col_name}')
    g.set_axis_labels('Month-Year', 'Quantity Sold')
    g.set_xticklabels(rotation=45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Sales Trends of Top Trending Products', fontsize=16)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'trending_products_quant_sold.png'))
    plt.show()

def find_customers_commerical(df2):
    df = df2.copy()

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')
    df = df[df['Quantity'] > 0]

    # Step 1: Identify customers with large purchase volumes
    large_volume_customers = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    large_volume_customers = large_volume_customers[
        large_volume_customers['Quantity'] > 1000]  # Threshold for large volumes

    # Step 2: Identify customers with high frequency of purchases
    purchase_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    purchase_frequency.columns = ['CustomerID', 'Frequency']
    high_frequency_customers = purchase_frequency[purchase_frequency['Frequency'] > 20]  # Threshold for high frequency

    # Step 3: Identify customers with high total spending
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    high_spending_customers = df.groupby('CustomerID')['TotalSpend'].sum().reset_index()
    high_spending_customers = high_spending_customers[
        high_spending_customers['TotalSpend'] > 10000]  # Threshold for high spending

    # Step 4: Combine all criteria to find potential commercial customers
    potential_commercial_customers = large_volume_customers.merge(
        high_frequency_customers, on='CustomerID').merge(
        high_spending_customers, on='CustomerID')

    potential_commercial_customers = potential_commercial_customers.sort_values(by='Quantity', ascending=False)

    top_25_customers = potential_commercial_customers.nlargest(25, 'Quantity')
    df_top_25_customers = df[df['CustomerID'].isin(top_25_customers['CustomerID'])]
    monthly_sales = df_top_25_customers.groupby(['CustomerID', 'MonthYear'])['Quantity'].sum().reset_index()
    monthly_sales['MonthYear'] = monthly_sales['MonthYear'].dt.to_timestamp()
    g = sns.FacetGrid(monthly_sales, col="CustomerID", col_wrap=5, height=3, aspect=1.5, sharey=False)
    g.map(sns.lineplot, 'MonthYear', 'Quantity')
    g.set_titles(col_template='Customer {col_name}')
    g.set_axis_labels('Month-Year', 'Quantity Purchased')
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Monthly Purchase Quantities for Top 25 Potential Commercial Customers', fontsize=16)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'commerical_customer.png'))
    plt.show()


def find_commerical_declining_customers(df2, calculation_method='streak-magnitude-log'):
    '''

    :param df2:
    :param calculation_method:
        streak-magnitude-log: find the negative streak and magnitude and take log of it
        if something else is passed: use only the negative streak count
    :return:
    '''
    df = df2.copy()

    # Convert InvoiceDate to datetime and extract month-year
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')

    # Step 1: Aggregate the monthly net quantities for each customer
    monthly_sales = df.groupby(['CustomerID', 'MonthYear'])['Quantity'].sum().reset_index()

    # Convert MonthYear back to datetime for plotting and analysis
    monthly_sales['MonthYear'] = monthly_sales['MonthYear'].dt.to_timestamp()

    # Step 2: Calculate the month-over-month percentage change in quantity for each customer
    monthly_sales['PercentageChange'] = monthly_sales.groupby('CustomerID')['Quantity'].pct_change() * 100

    # Fill NaN values with 0 for percentage change
    monthly_sales['PercentageChange'].fillna(0, inplace=True)

    # Step 3: Identify consecutive negative percentage changes and their magnitudes for each customer
    def consecutive_negatives_with_magnitude(series):
        max_streak = 0
        current_streak = 0
        magnitude_sum = 0
        max_magnitude_sum = 0
        for value in series:
            if value < 0:
                current_streak += 1
                magnitude_sum += abs(value)
                if current_streak > max_streak or (current_streak == max_streak and magnitude_sum > max_magnitude_sum):
                    max_streak = current_streak
                    max_magnitude_sum = magnitude_sum
            else:
                current_streak = 0
                magnitude_sum = 0
        return max_streak, max_magnitude_sum

    streaks = monthly_sales.groupby('CustomerID')['PercentageChange'].apply(
        consecutive_negatives_with_magnitude).reset_index()
    streaks.columns = ['CustomerID', 'StreakData']
    streaks['MaxConsecutiveNegatives'] = streaks['StreakData'].apply(lambda x: x[0])
    streaks['MagnitudeSum'] = streaks['StreakData'].apply(lambda x: x[1])

    # Step 4: Rank customers based on the longest and most significant negative streaks
    if calculation_method == 'streak-magnitude-log':
        streaks['Score'] = streaks['MaxConsecutiveNegatives'] * np.log(streaks['MagnitudeSum'])
    else:
        streaks['Score'] = streaks['MaxConsecutiveNegatives']
    top_25_declining_customers = streaks.nlargest(25, 'Score')

    # Step 5: Visualize the trends for these top 25 customers
    top_25_declining_sales = monthly_sales[monthly_sales['CustomerID'].isin(top_25_declining_customers['CustomerID'])]

    # Create the profile plot using FacetGrid
    g = sns.FacetGrid(top_25_declining_sales, col="CustomerID", col_wrap=5, height=3, aspect=1.5, sharey=False)
    g.map(sns.lineplot, 'MonthYear', 'Quantity')
    g.set_titles(col_template='Customer {col_name}')
    g.set_axis_labels('Month-Year', 'Quantity Purchased')
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Monthly Purchase Quantities for Top 25 Customers with Longest and Most Significant Declines',
                   fontsize=16)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'commerical_customers_declining.png'))
    plt.show()


def find_commercial_booming_customers(df2):
    df = df2.copy()
    # Convert InvoiceDate to datetime and extract month-year
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M')

    # Step 1: Identify customers with large purchase volumes
    large_volume_customers = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    large_volume_customers = large_volume_customers[
        large_volume_customers['Quantity'] > 1000]  # Threshold for large volumes

    # Step 2: Identify customers with high frequency of purchases
    purchase_frequency = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    purchase_frequency.columns = ['CustomerID', 'Frequency']
    high_frequency_customers = purchase_frequency[purchase_frequency['Frequency'] > 20]  # Threshold for high frequency

    # Step 3: Identify customers with high total spending
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    high_spending_customers = df.groupby('CustomerID')['TotalSpend'].sum().reset_index()
    high_spending_customers = high_spending_customers[
        high_spending_customers['TotalSpend'] > 10000]  # Threshold for high spending

    # Step 4: Combine all criteria to find potential commercial customers
    potential_commercial_customers = large_volume_customers.merge(
        high_frequency_customers, on='CustomerID').merge(
        high_spending_customers, on='CustomerID')

    # Step 5: Aggregate the monthly purchase quantities for these potential commercial customers
    df_potential_customers = df[df['CustomerID'].isin(potential_commercial_customers['CustomerID'])]
    monthly_sales = df_potential_customers.groupby(['CustomerID', 'MonthYear'])['Quantity'].sum().reset_index()

    # Convert MonthYear back to datetime for plotting and analysis
    monthly_sales['MonthYear'] = monthly_sales['MonthYear'].dt.to_timestamp()

    # Step 6: Calculate the trend (slope) of monthly purchases for each customer
    def calculate_slope(customer_data):
        customer_data = customer_data.reset_index(drop=True)
        X = np.arange(len(customer_data)).reshape(-1, 1)  # Use the index as the time variable
        y = customer_data['Quantity'].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]  # Return the slope (coefficient)

    slopes = monthly_sales.groupby('CustomerID').apply(calculate_slope).reset_index()
    slopes.columns = ['CustomerID', 'Slope']

    # Step 7: Identify customers with a positive slope (indicating an increasing trend)
    booming_customers = slopes[slopes['Slope'] > 0]
    top_25_booming_customers = booming_customers.nlargest(25, 'Slope')

    # Step 9: Visualize the purchase trends for these top 25 booming businesses
    top_25_booming_sales = monthly_sales[monthly_sales['CustomerID'].isin(top_25_booming_customers['CustomerID'])]

    # Create the profile plot using FacetGrid
    g = sns.FacetGrid(top_25_booming_sales, col="CustomerID", col_wrap=5, height=3, aspect=1.5, sharey=False)
    g.map(sns.lineplot, 'MonthYear', 'Quantity')
    g.set_titles(col_template='Customer {col_name}')
    g.set_axis_labels('Month-Year', 'Quantity Purchased')
    for ax in g.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Monthly Purchase Quantities for Top 25 Booming Businesses', fontsize=16)
    plt.savefig(os.path.join(IMAGE_DIRNAME, 'commerial_customers_booming.png'))
    plt.show()


if __name__ == "__main__":
    filepath = r'C:\Users\milan\Documents\projects\retail-customer-analysis\data\Online Retail.csv'

    df = pd.read_csv(filepath)
    df = df[df['UnitPrice'] > 0]
    df = df[df['Country'] == 'United Kingdom']
    df['Transaction'] = df['Quantity'].apply(lambda x: 'Return' if x < 0 else 'Bought')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])  # Convert to datetime format
    df['Date'] = df['InvoiceDate'].dt.date
    df['Time'] = df['InvoiceDate'].dt.strftime('%H:%M:%S')
    df['Time'] = df['InvoiceDate'].dt.hour
    df['Time of Day'] = ['Early Morning' if x < 6 else ('Morning' if x < 12 else ('Noon' if x < 14 else
                                                       ('Afternoon' if x < 18 else ('Evening' if x < 22 else 'Night'))))
                                                       for x in df['Time']]

    print(df.describe())


    plot_transaction_type(df)
    plot_time_of_day_bar(df)
    top_invoice_customer_timings(df)
    create_description_wordcloud(df)
    top_product_descriptions(df)
    make_trending_products(df)
    find_customers_commerical(df)
    find_commerical_declining_customers(df)
    find_commercial_booming_customers(df)
