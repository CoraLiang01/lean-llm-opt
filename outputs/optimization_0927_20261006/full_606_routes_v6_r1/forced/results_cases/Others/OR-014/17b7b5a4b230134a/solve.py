import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].str.strip()
df['Revenue'] = df['Revenue'].astype(float)
df['Demand'] = df['Demand'].astype(int)
df['Initial Inventory'] = df['Initial Inventory'].astype(int)
product_ids = df['Product Name'].tolist()
revenue = dict(zip(df['Product Name'], df['Revenue']))
demand = dict(zip(df['Product Name'], df['Demand']))
initial_inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
for pid in product_ids:
    if pid not in revenue or pid not in demand or pid not in initial_inventory:
        raise ValueError(f"Missing parameter for product '{pid}'.")
m = gp.Model('PizzaSalesRevenueMaximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= demand[pid] for pid in product_ids), name='')
m.addConstrs((x_vars[pid] <= initial_inventory[pid] for pid in product_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pid in product_ids:
        fulfilled = int(round(x_vars[pid].X))
        print(f'{pid}: Fulfilled {fulfilled} units (Demand: {demand[pid]}, Inventory: {initial_inventory[pid]}, Revenue/unit: {revenue[pid]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')