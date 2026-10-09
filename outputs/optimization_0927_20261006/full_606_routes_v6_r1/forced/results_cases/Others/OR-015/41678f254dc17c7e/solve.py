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
    raise ValueError("No products found with 'Product Name' starting with 'Aalop'.")
aalop_products = aalop_df['Product Name'].tolist()
revenue_dict = {}
demand_dict = {}
inventory_dict = {}
for (idx, row) in aalop_df.iterrows():
    pname = row['Product Name']
    try:
        revenue = int(row['Revenue'])
        demand = int(row['Demand'])
        inventory = float(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value for product '{pname}': {e}")
    revenue_dict[pname] = revenue
    demand_dict[pname] = demand
    inventory_dict[pname] = inventory
m = gp.Model('AalopProductRevenueMaximization')
x_vars = m.addVars(aalop_products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in aalop_products)), gp.GRB.MAXIMIZE)
for i in aalop_products:
    m.addConstr(x_vars[i] <= inventory_dict[i], name=f'inv_{i}')
    m.addConstr(x_vars[i] <= demand_dict[i], name=f'demand_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_revenue = m.objVal
    print(f'Optimal total revenue: {total_revenue:.2f}')
    print("--- Optimal Fulfillment Plan for 'Aalop' Products ---")
    for i in aalop_products:
        print(f'Product: {i}')
        print(f'  Units fulfilled (x): {x_vars[i].X:.0f}')
        print(f'  Revenue per unit: {revenue_dict[i]}')
        print(f'  Demand: {demand_dict[i]}')
        print(f'  Initial Inventory: {inventory_dict[i]:.0f}')
else:
    print(f'No optimal solution found. Status: {m.status}')