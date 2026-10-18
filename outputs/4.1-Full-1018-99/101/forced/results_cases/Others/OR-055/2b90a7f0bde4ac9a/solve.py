import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'DisplayID' and 'Capacity' columns.")
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
display_ids = capacity_df['DisplayID'].tolist()
if len(set(display_ids)) != len(display_ids):
    raise ValueError('Duplicate DisplayID found in capacity.csv.')
capacity_dict = {}
for idx, row in capacity_df.iterrows():
    did = row['DisplayID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for DisplayID {did}: {row['Capacity']}")
    capacity_dict[did] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
if len(set(product_names)) != len(product_names):
    raise ValueError('Duplicate ProductName found in products.csv.')
value_dict = {}
weight_dict = {}
for idx, row in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wt
I = display_ids
J = product_names
for did in I:
    if did not in capacity_dict:
        raise KeyError(f'Missing capacity for DisplayID {did}')
for pname in J:
    if pname not in value_dict or pname not in weight_dict:
        raise KeyError(f'Missing value or weight for ProductName {pname}')
m = gp.Model('BoatDisplayAllocation')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in J)) <= capacity_dict[i], name=f'cap_{i}')
m.optimize()