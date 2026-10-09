import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing 'Capacity' column in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Could not parse capacity value: {e}')
required_cols = ['ProductName', 'Value', 'Weight']
for col in required_cols:
    if col not in products_df.columns:
        raise KeyError(f"Missing '{col}' column in products.csv")
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('Duplicate ProductName entries found in products.csv')
try:
    value_dict = dict(zip(product_names, products_df['Value'].astype(int)))
    weight_dict = dict(zip(product_names, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Could not parse Value or Weight columns as integers: {e}')
for pname in product_names:
    if pname not in value_dict or pname not in weight_dict:
        raise ValueError(f"Missing Value or Weight for product '{pname}'")

def solve_problem(product_names, value_dict, weight_dict, capacity):
    m = gp.Model('PharmacyRestock')
    quantity_vars = m.addVars(product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * quantity_vars[i] for i in product_names)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * quantity_vars[i] for i in product_names)) <= capacity, name='capacity')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(product_names, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')