import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f
import matplotlib.pyplot as plt

# Create a numpy array for the X data (percentage of ADHD diagnosis)
X = np.array([[5.50], [5.90], [5.60], [6.90], [6.30], [7.20], [6.40], [7.40], [6.50], [7.40], [7.20], [8.10], [8.60], [8.40], [8.50], [9.50], [9.80], [10.30]])

# Create a numpy array for the Y data (annual costs of ADHD)
Y = np.array([2688.48, 2692.21, 2707.70, 2780.81, 2792.86, 2839.64, 2841.88, 2877.18, 2824.79, 2837.74, 3064.66, 3144.61, 3290.632, 3328.92, 3362.07, 3547.94, 3663.65, 3705.38])

# Create a linear regression object
model = LinearRegression()

# Fit the linear regression model
model.fit(X, Y)

# Plot the scatter plot of the data
plt.scatter(X, Y, color='blue', label='Data')

# Plot the line of best fit
plt.plot(X, model.predict(X), color='red', label='Linear Regression')

# Add labels and title to the plot
plt.xlabel('Percentage of ADHD Diagnosis')
plt.ylabel('Annual Costs of ADHD')
plt.title('Linear Regression Analysis')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Calculate the residual sum of squares (RSS)
RSS = np.sum((Y - model.predict(X)) ** 2)

# Calculate the total sum of squares (TSS)
TSS = np.sum((Y - np.mean(Y)) ** 2)

# Calculate the degrees of freedom for the model and error
n = len(X)
p = 1
df_model = p
df_error = n - p - 1

# Calculate the mean square for the model and error
MS_model = RSS / df_model
MS_error = TSS / df_error

# Calculate the F-statistic and p-value
F = MS_model / MS_error
p_value = f.sf(F, df_model, df_error)

# Print the ANOVA table
print('Source\t\tDF\tSum of Squares\tMean Square\tF-statistic\tp-value')
print('Model\t\t{}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}'.format(df_model, RSS, MS_model, F, p_value))
print('Error\t\t{}\t{:.4f}\t\t{:.4f}'.format(df_error, TSS - RSS, MS_error))
print('Total\t\t{}\t{:.4f}'.format(n - 1, TSS))

# Calculate the average income in America
average_income = 31133

def predict_adhd_growth(annual_costs, average_income):
    # Calculate the prediction for ADHD growth that would result in given annual costs as a percentage of average income
    prediction = (average_income * annual_costs - model.intercept_) / model.coef_

    return prediction[0]
# Calculate the average income in America
average_income = 31133

def predict_adhd_growth(annual_costs, average_income):
    # Calculate the prediction for ADHD growth that would result in given annual costs as a percentage of average income
    prediction = (average_income * annual_costs - model.intercept_) / model.coef_

    return prediction[0]

# Example usage: Predict the percentage of growth in ADHD diagnosis rates for annual costs of 10% of the average income in America
prediction_10_percent = predict_adhd_growth(0.1, average_income)
print("The predicted percentage of growth in ADHD diagnosis rates for annual costs of 10% of the average income in America is:", prediction_10_percent)

# Define the current ADHD diagnosis rate
current_diagnosis_rate = 10.30

# Calculate the predicted increase in ADHD diagnosis rate for each year
predicted_increase = current_diagnosis_rate * (prediction_10_percent / 100)

# Create a list to store the projected ADHD diagnosis rates
projected_diagnosis_rates = []

projected_diagnosis_rates = []

# Create a list to store the projected costs
projected_costs = []

# Generate the projected ADHD diagnosis rates and costs up to the year 2031
for year in range(2022, 2032):
    projected_diagnosis_rates.append(current_diagnosis_rate)
    projected_cost = model.predict([[current_diagnosis_rate]])
    projected_costs.append(projected_cost[0])
    current_diagnosis_rate += predicted_increase

# Print the projected ADHD diagnosis rates and costs up to the year 2031
print("Projected ADHD Diagnosis Rates and Costs:")
print("Year\t\tDiagnosis Rate\t\tProjected Cost")
for year, diagnosis_rate, cost in zip(range(2022, 2032), projected_diagnosis_rates, projected_costs):
    print(f"{year}\t\t{diagnosis_rate:.2f}%\t\t${cost:.2f}")
