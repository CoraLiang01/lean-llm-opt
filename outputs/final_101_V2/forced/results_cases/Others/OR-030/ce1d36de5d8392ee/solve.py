import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
df = pd.read_csv(csv_path, sep=',')
df['Product Name'] = df['Product Name'].astype(str).str.strip()
fdk57_df = df[df['Product Name'] == 'FDK57'].copy()
if fdk57_df.empty:
    raise ValueError("No car models with 'Product Name' exactly equal to 'FDK57' found in the data.")
model_keys = list(fdk57_df.index)
revenue = fdk57_df['Revenue'].to_dict()
demand = fdk57_df['Demand'].to_dict()
inventory = fdk57_df['Initial Inventory'].to_dict()
m = gp.Model('FDK57_Sales_MaxRevenue')
x = m.addVars(model_keys, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.addConstrs((x[i] <= demand[i] for i in model_keys), name='')
m.addConstrs((x[i] <= inventory[i] for i in model_keys), name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in model_keys)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- FDK57 Sales Fulfillment Plan ---')
    for i in model_keys:
        fulfilled = x[i].X
        print(f'Row {i}: FDK57 | Revenue/unit: {revenue[i]:.4f} | Demand: {demand[i]} | Inventory: {inventory[i]} | Fulfilled: {fulfilled:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')