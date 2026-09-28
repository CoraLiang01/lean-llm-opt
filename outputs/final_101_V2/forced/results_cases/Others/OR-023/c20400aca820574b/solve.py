import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
df = pd.read_csv(csv_path, sep=',')
ele_s_mask = df['Product_Reference'].astype(str).str.contains('ELE-S', case=False, regex=False)
ele_s_df = df[ele_s_mask].copy()
products = ele_s_df['Product_Reference'].astype(str).tolist()
required_cols = ['Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if ele_s_df[col].isnull().any():
        raise ValueError(f"Missing data in column '{col}' for some ELE-S products.")
revenue = ele_s_df.set_index('Product_Reference')['Revenue'].to_dict()
demand = ele_s_df.set_index('Product_Reference')['Demand'].to_dict()
init_inventory = ele_s_df.set_index('Product_Reference')['Initial Inventory'].to_dict()
max_fulfill = {}
for p in products:
    d = float(demand[p])
    inv = float(init_inventory[p])
    max_fulfill[p] = min(d, inv)
m = gp.Model('ELE-S_Revenue_Maximization')
x = m.addVars(products, lb=0.0, ub=[max_fulfill[p] for p in products], vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[p] * x[p] for p in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Fulfillment Plan for ELE-S Products ---')
    for p in products:
        fulfilled = x[p].X
        print(f'Product {p}: Fulfilled {fulfilled:.2f} units (Revenue per unit: {revenue[p]:.2f}, Demand: {demand[p]}, Initial Inventory: {init_inventory[p]})')
else:
    print(f'No optimal solution found. Status: {m.status}')