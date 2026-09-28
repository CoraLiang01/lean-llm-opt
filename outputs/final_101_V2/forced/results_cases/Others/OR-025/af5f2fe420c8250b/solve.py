import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
df = pd.read_csv(csv_path, sep=',')
is_tablet = df['Product Name'].astype(str).str.strip().str.casefold().str.startswith('tablet')
tablets_df = df[is_tablet].copy()
if tablets_df.empty:
    raise ValueError('No TABLET models found in the data.')
tablet_ids = tablets_df['Product Name'].astype(str).tolist()
revenue = tablets_df.set_index('Product Name')['Revenue'].astype(float).to_dict()
demand = tablets_df.set_index('Product Name')['Demand'].astype(int).to_dict()
inventory = tablets_df.set_index('Product Name')['Initial Inventory'].astype(int).to_dict()
fulfill_limit = {i: min(demand[i], inventory[i]) for i in tablet_ids}
for i in tablet_ids:
    if i not in revenue or i not in demand or i not in inventory:
        raise KeyError(f"Missing data for TABLET model '{i}'.")
    if fulfill_limit[i] < 0:
        raise ValueError(f"Negative fulfillment limit for TABLET model '{i}'.")
m = gp.Model('TabletRevenueMaximization')
x = m.addVars(tablet_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x[i] for i in tablet_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x[i] <= inventory[i] for i in tablet_ids), name='')
m.addConstrs((x[i] <= demand[i] for i in tablet_ids), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan for TABLET Models ---')
    for i in tablet_ids:
        fulfilled = int(round(x[i].X))
        print(f'Model: {i}')
        print(f'  Units fulfilled: {fulfilled}')
        print(f'  Revenue per unit: {revenue[i]:.2f}')
        print(f'  Demand: {demand[i]}')
        print(f'  Initial Inventory: {inventory[i]}')
        print(f'  Fulfillment upper bound: {fulfill_limit[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')