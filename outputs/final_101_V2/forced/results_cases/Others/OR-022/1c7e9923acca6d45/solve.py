import gurobipy as gp
import pandas as pd
import numpy as np
import re
salesorders_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
df = pd.read_csv(salesorders_path, sep=',')

def is_27in(name):
    return bool(re.search('\\b27in\\b', str(name).replace(' ', ''), re.IGNORECASE)) or '27in' in str(name).replace(' ', '')
mask_27in = df['Product Name'].str.replace(' ', '', regex=False).str.contains('27in', case=False, regex=False)
df_27in = df[mask_27in].copy()
if df_27in.empty:
    raise ValueError("No products with '27in' in the 'Product Name' found in the data.")
products = df_27in['Product Name'].tolist()
revenue = df_27in.set_index('Product Name')['Revenue'].to_dict()
demand = df_27in.set_index('Product Name')['Demand'].to_dict()
inventory = df_27in.set_index('Product Name')['Initial Inventory'].to_dict()
for pname in products:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('27in_Fulfillment')
x = m.addVars(products, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
for pname in products:
    m.addConstr(x[pname] <= float(demand[pname]), name=f'demand_{pname}')
    m.addConstr(x[pname] <= float(inventory[pname]), name=f'inventory_{pname}')
    m.addConstr(x[pname] >= 0.0, name=f'nonneg_{pname}')
m.setObjective(gp.quicksum((revenue[pname] * x[pname] for pname in products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.4f}')
    print("--- Fulfillment Plan for '27in' Products ---")
    for pname in products:
        print(f'Product: {pname}')
        print(f'  Fulfilled units (x): {x[pname].X:.4f}')
        print(f'  Demand: {demand[pname]}')
        print(f'  Initial Inventory: {inventory[pname]}')
        print(f'  Per-unit Revenue: {revenue[pname]:.4f}')
else:
    print(f'No optimal solution found. Status: {m.status}')