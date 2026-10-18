import gurobipy as gp
import pandas as pd
import numpy as np
import re
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'CabinetID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'CabinetID' and 'Capacity'.")
capacity_df['CabinetID'] = capacity_df['CabinetID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
cabinet_ids = capacity_df['CabinetID'].astype(int).tolist()
cabinet_capacities = dict(zip(capacity_df['CabinetID'].astype(int), capacity_df['Capacity'].astype(float)))
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
product_values = dict(zip(products_df['ProductName'], products_df['Value'].astype(float)))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight'].astype(float)))
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinet_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for i in cabinet_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in cabinet_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= cabinet_capacities[i], name=f'cabinet_capacity_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.2f}')
    print('--- Allocation Plan (cabinet, product, units) ---')
    for i in cabinet_ids:
        for j in product_names:
            val = x_vars[i, j].X
            if val > 1e-06:
                print(f"Cabinet {i} - Product '{j}': {int(round(val))} units")
else:
    print(f'No optimal solution found. Status: {m.status}')