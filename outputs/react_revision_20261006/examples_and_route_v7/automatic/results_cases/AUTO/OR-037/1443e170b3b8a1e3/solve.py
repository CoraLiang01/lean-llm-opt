import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
product_ids = products_df['ProductName'].astype(str).tolist()
if not set(['Value', 'Weight']).issubset(products_df.columns):
    raise KeyError("Missing required columns 'Value' or 'Weight' in products.csv")
try:
    value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
    weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'Value' or 'Weight' columns to int: {e}")
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing required column 'Capacity' in capacity.csv")
try:
    capacity_val = int(capacity_df['Capacity'].iloc[0])
except Exception as e:
    raise ValueError(f"Error converting 'Capacity' to int: {e}")
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise KeyError(f"Missing value or weight for product '{pid}'")

def solve_problem(product_ids, value_dict, weight_dict, capacity_val):
    m = gp.Model('CarSalesInventory')
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[pid] * x_vars[pid] for pid in product_ids)) <= capacity_val, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(product_ids, value_dict, weight_dict, capacity_val)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')