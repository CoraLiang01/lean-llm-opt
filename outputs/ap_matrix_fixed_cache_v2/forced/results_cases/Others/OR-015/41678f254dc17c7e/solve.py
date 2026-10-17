import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def is_aalop(name):
    return name.strip().casefold().startswith('aalop')
aalop_mask = df['Product Name'].apply(is_aalop)
aalop_df = df[aalop_mask].copy()
if aalop_df.empty:
    raise ValueError("No products found with names starting with 'Aalop'.")
aalop_products = aalop_df['Product Name'].tolist()

def to_int_series(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
revenue = to_int_series(aalop_df['Revenue'], 'Revenue')
demand = to_int_series(aalop_df['Demand'], 'Demand')
initial_inventory = to_float_series(aalop_df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(aalop_products, revenue))
demand_dict = dict(zip(aalop_products, demand))
inventory_dict = dict(zip(aalop_products, initial_inventory))
m = gp.Model('Aalop_Product_Revenue_Maximization')
x_vars = m.addVars(aalop_products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in aalop_products)), gp.GRB.MAXIMIZE)
for i in aalop_products:
    m.addConstr(x_vars[i] <= inventory_dict[i], name=f'inv_{i}')
    m.addConstr(x_vars[i] <= demand_dict[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Aalop Product Fulfillment Plan ---')
    for i in aalop_products:
        print(f'Product: {i} | Units fulfilled: {int(round(x_vars[i].X))} | Revenue per unit: {revenue_dict[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')