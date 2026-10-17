import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_names = df['Product Name'].astype(str).tolist()
df = df.set_index('Product Name', drop=False)

def safe_int_convert(series, colname):
    try:
        return series.astype(str).str.strip().astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue = safe_int_convert(df['Revenue'], 'Revenue').to_dict()
demand = safe_int_convert(df['Demand'], 'Demand').to_dict()
initial_inventory = safe_int_convert(df['Initial Inventory'], 'Initial Inventory').to_dict()
for pname in product_names:
    if pname not in revenue or pname not in demand or pname not in initial_inventory:
        raise KeyError(f"Missing parameter(s) for product '{pname}'.")
m = gp.Model('RetailMerchandiseAllocation')
x_vars = m.addVars(product_names, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[pname] * x_vars[pname] for pname in product_names)), gp.GRB.MAXIMIZE)
for pname in product_names:
    m.addConstr(x_vars[pname] <= demand[pname], name=f'demand_{pname}')
    m.addConstr(x_vars[pname] <= initial_inventory[pname], name=f'inventory_{pname}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Optimal Fulfillment Plan ---')
    for pname in product_names:
        qty = x_vars[pname].X
        if qty > 1e-06:
            print(f'{pname}: Fulfill {qty:.2f} units (Revenue per unit: {revenue[pname]}, Demand: {demand[pname]}, Inventory: {initial_inventory[pname]})')
else:
    print(f'No optimal solution found. Status: {m.status}')