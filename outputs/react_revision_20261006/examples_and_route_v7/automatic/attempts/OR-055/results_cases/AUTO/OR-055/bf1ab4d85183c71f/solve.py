import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
display_ids = capacity_df['DisplayID'].astype(str).tolist()
display_capacities = {}
for (idx, row) in capacity_df.iterrows():
    did = str(row['DisplayID'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f'Invalid Capacity value for DisplayID {did}')
    display_capacities[did] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
product_names = products_df['ProductName'].astype(str).tolist()
product_values = {}
product_weights = {}
for (idx, row) in products_df.iterrows():
    pname = str(row['ProductName'])
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Invalid Value or Weight for ProductName {pname}')
    product_values[pname] = val
    product_weights[pname] = wt
if set(display_ids) != set(display_capacities.keys()):
    raise ValueError('Mismatch in display area identifiers and capacities')
if set(product_names) != set(product_values.keys()) or set(product_names) != set(product_weights.keys()):
    raise ValueError('Mismatch in product identifiers and value/weight parameters')
decision_keys = [(i, j) for i in display_ids for j in product_names]
m = gp.Model('BoatDisplayAllocation')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[j] * x_vars[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((product_weights[j] * x_vars[i, j] for j in product_names)) <= display_capacities[i], name=f'cap_{i}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for (i, j) in decision_keys:
        var = x_vars[i, j]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')