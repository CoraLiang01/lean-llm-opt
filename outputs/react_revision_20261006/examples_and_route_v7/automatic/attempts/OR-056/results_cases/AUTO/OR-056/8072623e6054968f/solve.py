import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError('Missing required columns in capacity.csv')
display_ids = capacity_df['DisplayID'].astype(int).tolist()
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    did = int(row['DisplayID'])
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Non-integer Capacity for DisplayID {row['DisplayID']}")
    if did in capacity_dict:
        raise ValueError(f'Duplicate DisplayID {did} in capacity.csv')
    capacity_dict[did] = cap
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError('Missing required columns in products.csv')
product_names = products_df['ProductName'].tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f'Non-integer Value or Weight for ProductName {pname}')
    if pname in value_dict or pname in weight_dict:
        raise ValueError(f'Duplicate ProductName {pname} in products.csv')
    value_dict[pname] = val
    weight_dict[pname] = wt
if set(display_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch in display area identifiers')
if set(product_names) != set(value_dict.keys()) or set(product_names) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers')
decision_keys = [(i, j) for i in display_ids for j in product_names]

def solve_boat_display_assignment(display_ids, product_names, capacity_dict, value_dict, weight_dict, decision_keys):
    m = gp.Model('BoatDisplayAssignment')
    x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for (i, j) in decision_keys)), gp.GRB.MAXIMIZE)
    for i in display_ids:
        m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name='cap_%d' % i)
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_boat_display_assignment(display_ids, product_names, capacity_dict, value_dict, weight_dict, decision_keys)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')