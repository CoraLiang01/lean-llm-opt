import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def is_tablet(name):
    return name.strip().casefold().startswith('tablet')
tablet_mask = df['Product Name'].apply(is_tablet)
tablet_df = df[tablet_mask].copy()
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in tablet_df.columns:
        raise KeyError(f"Required column '{col}' not found in the data.")
tablet_df['Product Name'] = tablet_df['Product Name'].apply(lambda x: x.strip())
tablet_df.set_index('Product Name', inplace=True)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    tablet_df[col] = pd.to_numeric(tablet_df[col], errors='raise')
tablet_models = list(tablet_df.index)
revenue = tablet_df['Revenue'].to_dict()
demand = tablet_df['Demand'].to_dict()
initial_inventory = tablet_df['Initial Inventory'].to_dict()
fulfillment_ub = {i: min(demand[i], initial_inventory[i]) for i in tablet_models}
m = gp.Model('TabletRevenueMaximization')
x_vars = m.addVars(tablet_models, vtype=gp.GRB.INTEGER, lb=0, ub=[fulfillment_ub[i] for i in tablet_models], name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in tablet_models)), gp.GRB.MAXIMIZE)
for i in tablet_models:
    m.addConstr(x_vars[i] <= demand[i], name=f'demand_{i}')
    m.addConstr(x_vars[i] <= initial_inventory[i], name=f'inventory_{i}')
    m.addConstr(x_vars[i] >= 0, name=f'nonneg_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for TABLET Models ---')
    for i in tablet_models:
        fulfilled = int(round(x_vars[i].X))
        print(f'Model: {i}, Fulfilled: {fulfilled}, Demand: {demand[i]}, Inventory: {initial_inventory[i]}, Revenue/unit: {revenue[i]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')