
import pandas as pd

data = pd.read_csv(r'C:\Users\ITSME\Desktop\CI Project\Ecommerce_Delivery_Analytics_New.csv')
print("Delivery Delay vs Refund:")
print(pd.crosstab(data['Delivery Delay'], data['Refund Requested']))

print("\nService Rating vs Refund:")
print(pd.crosstab(data['Service Rating'], data['Refund Requested']))

