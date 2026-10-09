import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
products = df['Product Name'].tolist()
try:
    revenue = dict(zip(products, df['Revenue'].astype(int)))
    demand = dict(zip(products, df['Demand'].astype(int)))
    initial_inventory = dict(zip(products, df['Initial Inventory'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting numeric columns: {e}')
for p in products:
    if p not in revenue or p not in demand or p not in initial_inventory:
        raise ValueError(f"Missing parameter for product '{p}'.")
m = gp.Model('DemandFulfillment')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= initial_inventory[p] for p in products), name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()