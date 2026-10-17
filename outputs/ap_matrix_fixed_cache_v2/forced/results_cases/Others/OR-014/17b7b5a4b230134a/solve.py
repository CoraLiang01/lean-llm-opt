import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
pizza_ids = df['Product Name'].astype(str).tolist()
if not all((col in df.columns for col in ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])):
    raise KeyError('One or more required columns are missing from the CSV.')
df = df.set_index('Product Name', drop=False)
revenue = {}
demand = {}
init_inventory = {}
for pid in pizza_ids:
    row = df.loc[pid]
    try:
        revenue[pid] = float(row['Revenue'])
        demand[pid] = int(row['Demand'])
        init_inventory[pid] = int(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value for pizza '{pid}': {e}")
if not set(revenue.keys()) == set(demand.keys()) == set(init_inventory.keys()) == set(pizza_ids):
    raise ValueError('Mismatch in pizza identifier coverage among parameters.')
m = gp.Model('Pizza_Fulfillment_MaxRevenue')
x_vars = m.addVars(pizza_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in pizza_ids)), gp.GRB.MAXIMIZE)
m.addConstrs((x_vars[pid] <= demand[pid] for pid in pizza_ids), name='')
m.addConstrs((x_vars[pid] <= init_inventory[pid] for pid in pizza_ids), name='')
m.optimize()