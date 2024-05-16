import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths for each dataset
file_paths = {
    'Daily Data': "C:\\Users\\HP\\Downloads\\climate\\daily_data.csv",
    'Hourly Data': "C:\\Users\\HP\\Downloads\\climate\\hourly_data.csv",
    'Monthly Data': "C:\\Users\\HP\\Downloads\\climate\\monthly_data.csv",
    'Three-Hour Data': "C:\\Users\\HP\\Downloads\\climate\\three_hour_data.csv"
}

# Step 1: Data Collection
# Load each dataset
datasets = {data_type: pd.read_csv(path) for data_type, path in file_paths.items()}

# Step 2: Visualization
# Scatter plot for daily data with different marker and color
plt.figure(figsize=(10, 6))
plt.scatter(datasets['Daily Data']['DATE'], datasets['Daily Data']['DailyDepartureFromNormalAverageTemperature'], marker='x', color='orange')
plt.title('Daily Temperature Variation')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# KDE plot for hourly data with a different shade of blue
plt.figure(figsize=(10, 6))
sns.kdeplot(datasets['Hourly Data']['HourlyWetBulbTemperature'], color='skyblue', shade=True)
plt.title('Temperature Distribution (Hourly)')
plt.xlabel('Wet Bulb Temperature (°C)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Line plot for monthly data with a different line style
plt.figure(figsize=(10, 6))
plt.plot(datasets['Monthly Data']['DATE'], datasets['Monthly Data']['MonthlyDepartureFromNormalHeatingDegreeDays'], color='green', linestyle='--')
plt.title('Monthly Heating Degree Days Trend')
plt.xlabel('Date')
plt.ylabel('Heating Degree Days')
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# KDE plot for three-hour data with a different shade of red
plt.figure(figsize=(10, 6))
sns.kdeplot(datasets['Three-Hour Data']['HourlyPressureChange'], color='salmon', shade=True)
plt.title('Pressure Change Distribution (Three-Hourly)')
plt.xlabel('Pressure Change')
plt.ylabel('Density')
plt.grid(True)
plt.show()

