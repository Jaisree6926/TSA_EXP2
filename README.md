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

A - LINEAR TREND ESTIMATION

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Load the data from the CSV file
            data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\9. Sales-Data-Analysis.csv")
            
            # Parse the Date column
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
            
            # Aggregate quantity per actual date (daily)
            daily_data = data.groupby('Date')['Quantity'].sum().reset_index()
            
            # Prepare data for linear regression trend by date converted to fractional year
            years_fractional = daily_data['Date'].dt.year + (daily_data['Date'].dt.dayofyear - 1) / 365.25
            years = years_fractional.tolist()
            quantities = daily_data['Quantity'].tolist()
            
            X = [y - years[len(years)//2] for y in years]
            xy = [x*y for x, y in zip(X, quantities)]
            x2 = [x**2 for x in X]
            n = len(years)
            b = (n*sum(xy) - sum(quantities)*sum(X)) / (n*sum(x2) - sum(X)**2)
            a = (sum(quantities) - b*sum(X)) / n
            linear_trend = [a + b*x for x in X]
            
            daily_data['Linear Trend'] = linear_trend
            
            # Set Date as index for plotting
            daily_data.set_index('Date', inplace=True)
            
            # Plot daily quantity and linear trend with date x-axis
            plt.figure(figsize=(14, 6))
            plt.plot(daily_data.index, daily_data['Quantity'], marker='o', label='Actual Quantity')
            plt.plot(daily_data.index, daily_data['Linear Trend'], linestyle='--', label='Linear Trend')
            plt.xlabel('Date')
            plt.ylabel('Quantity Sold')
            plt.title('Daily Sales Quantity with Linear Trend')
            plt.legend()
            plt.grid(True)
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

B- POLYNOMIAL TREND ESTIMATION

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # Load and preprocess sales data
            data = pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\9. Sales-Data-Analysis.csv")
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')  # Convert to datetime
            
            # Aggregate Quantity by actual Date (daily)
            daily_data = data.groupby('Date')['Quantity'].sum().reset_index()
            
            # Prepare X (time index centered) and y (quantities)
            years_fractional = daily_data['Date'].dt.year + (daily_data['Date'].dt.dayofyear - 1) / 365.25
            years = years_fractional.tolist()
            quantities = daily_data['Quantity'].tolist()
            
            # Center X around middle value to improve numerical stability
            X = [year - years[len(years) // 2] for year in years]
            x2 = [x_i ** 2 for x_i in X]
            xy = [x_i * y_i for x_i, y_i in zip(X, quantities)]
            n = len(years)
            denominator = n * sum(x2) - (sum(X) ** 2)
            if denominator == 0:
                raise ValueError("No variation in X values to fit polynomial trend.")
            
            # Polynomial trend (degree 2) calculations
            x3 = [x_i ** 3 for x_i in X]
            x4 = [x_i ** 4 for x_i in X]
            x2y = [x2_i * y_i for x2_i, y_i in zip(x2, quantities)]
            coeff_matrix = np.array([
                [n,         sum(X),     sum(x2)],
                [sum(X),    sum(x2),    sum(x3)],
                [sum(x2),   sum(x3),    sum(x4)]
            ])
            Y = np.array([
                sum(quantities),
                sum(xy),
                sum(x2y)
            ])
            
            # Solve for polynomial coefficients a_poly, b_poly, c_poly
            solution = np.linalg.solve(coeff_matrix, Y)
            a_poly, b_poly, c_poly = solution
            poly_trend = [a_poly + b_poly * X[i] + c_poly * x2[i] for i in range(n)]
            
            # Print polynomial equation
            print(f"Polynomial Trend: y = {a_poly:.2f} + {b_poly:.2f}x + {c_poly:.2f}x²")
            
            # Prepare DataFrame for plotting
            daily_data['Polynomial Trend'] = poly_trend
            daily_data.set_index('Date', inplace=True)
            
            # Plot actual daily data and polynomial trend line with date x-axis
            plt.figure(figsize=(14, 6))
            plt.plot(daily_data.index, daily_data['Quantity'], marker='o', linestyle='-', color='blue', label='Actual Quantity')
            plt.plot(daily_data.index, daily_data['Polynomial Trend'], linestyle='-', color='red', label='Polynomial Trend')
            plt.xlabel('Date')
            plt.ylabel('Total Quantity Sold')
            plt.title('Daily Sales Quantity with Polynomial Trend')
            plt.legend()
            plt.grid(True)
            
            # Set x-axis locator and formatter to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()


### OUTPUT

A - LINEAR TREND ESTIMATION


<img width="1406" height="579" alt="Screenshot 2025-08-28 153906" src="https://github.com/user-attachments/assets/52226ead-46ea-441d-b468-43720d79bdff" />


B- POLYNOMIAL TREND ESTIMATION


<img width="1423" height="576" alt="Screenshot 2025-08-28 154442" src="https://github.com/user-attachments/assets/138dedbc-5154-4c90-b971-46958c8e9ec6" />


### RESULT:
Thus the Python program for linear and Polynomial Trend Estiamtion has been executed successfully.
