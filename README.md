# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:

      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      
      # Load and preprocess sales data
      data = pd.read_csv(r'C:\Users\admin\OneDrive\Desktop\9. Sales-Data-Analysis.csv')
      data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
      
      # Aggregate Quantity by Month
      data['Month'] = data['Date'].dt.to_period('M')
      monthly_data = data.groupby('Month')['Quantity'].sum().reset_index()
      monthly_data['Month'] = monthly_data['Month'].dt.to_timestamp()
      
      # Prepare centered time index X and y quantities
      years_fractional = monthly_data['Month'].dt.year + (monthly_data['Month'].dt.month - 1) / 12
      years = years_fractional.tolist()
      quantities = monthly_data['Quantity'].tolist()
      
      X = [y - years[len(years) // 2] for y in years]
      x2 = [x ** 2 for x in X]
      
      n = len(years)
      xy = [x * y for x, y in zip(X, quantities)]
      x3 = [x ** 3 for x in X]
      x4 = [x ** 4 for x in X]
      x2y = [x2_i * y for x2_i, y in zip(x2, quantities)]
      
      # Linear trend coefficients
      b_linear = (n * sum(xy) - sum(X) * sum(quantities)) / (n * sum(x2) - (sum(X)) ** 2)
      a_linear = (sum(quantities) - b_linear * sum(X)) / n
      linear_trend = [a_linear + b_linear * x for x in X]
      
      # Polynomial trend coefficients (degree 2)
      coeff_matrix = np.array([
          [n, sum(X), sum(x2)],
          [sum(X), sum(x2), sum(x3)],
          [sum(x2), sum(x3), sum(x4)]
      ])
      Y = np.array([sum(quantities), sum(xy), sum(x2y)])
      a_poly, b_poly, c_poly = np.linalg.solve(coeff_matrix, Y)
      poly_trend = [a_poly + b_poly * X[i] + c_poly * x2[i] for i in range(n)]
      
      # Add trend columns to DataFrame
      monthly_data['Linear Trend'] = linear_trend
      monthly_data['Polynomial Trend'] = poly_trend
      
      # Plot 1: Linear Trend Estimation
      plt.figure(figsize=(12, 5))
      plt.plot(monthly_data['Month'], monthly_data['Quantity'], marker='o', label='Actual Quantity', color='blue')
      plt.plot(monthly_data['Month'], monthly_data['Linear Trend'], linestyle='--', label='Linear Trend', color='black')
      plt.xlabel('Month')
      plt.ylabel('Total Quantity Sold')
      plt.title('Monthly Sales Quantity with Linear Trend')
      plt.legend()
      plt.grid(True)
      plt.show()
      
      # Plot 2: Polynomial Trend Estimation (Degree 2)
      plt.figure(figsize=(12, 5))
      plt.plot(monthly_data['Month'], monthly_data['Quantity'], marker='o', label='Actual Quantity', color='blue')
      plt.plot(monthly_data['Month'], monthly_data['Polynomial Trend'], linestyle='-', label='Polynomial Trend (Degree 2)', color='red')
      plt.xlabel('Month')
      plt.ylabel('Total Quantity Sold')
      plt.title('Monthly Sales Quantity with Polynomial Trend (Degree 2)')
      plt.legend()
      plt.grid(True)
      plt.show()

### OUTPUT

A - LINEAR TREND ESTIMATION

<img width="1308" height="580" alt="Screenshot 2025-08-26 105332" src="https://github.com/user-attachments/assets/e927f3ad-dbb2-49e4-9186-7cd017b07a8a" />


B- POLYNOMIAL TREND ESTIMATION

<img width="1346" height="583" alt="Screenshot 2025-08-26 105401" src="https://github.com/user-attachments/assets/1806a21c-1bb0-4a0f-bcdc-5b1fd7cf37ea" />


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
