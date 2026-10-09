import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['Product Name'] = df['Product Name'].astype(str)
df = df.set_index('Product Name', drop=False)
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    df[col] = df[col].str.strip()
    if col == 'Revenue':
        df[col] = df[col].astype(float)
    else:
        df[col] = df[col].astype(int)
products = list(df.index)
revenue = df['Revenue'].to_dict()
demand = df['Demand'].to_dict()
inventory = df['Initial Inventory'].to_dict()
for i in products:
    if not (np.isfinite(revenue[i]) and isinstance(demand[i], (int, np.integer)) and isinstance(inventory[i], (int, np.integer))):
        raise ValueError(f"Non-numeric or missing parameter for product '{i}'.")
m = gp.Model('MaximizeRevenue')
x_vars = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
demand_constrs = m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
inventory_constrs = m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), gp.GRB.MAXIMIZE)
m.optimize()