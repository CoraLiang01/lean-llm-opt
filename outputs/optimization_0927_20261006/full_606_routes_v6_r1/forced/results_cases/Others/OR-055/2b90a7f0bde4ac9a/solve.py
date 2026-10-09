import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'DisplayID' and 'Capacity' columns.")
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
display_ids = capacity_df['DisplayID'].tolist()
display_ids_int = [int(i) for i in display_ids]
display_id_map = dict(zip(display_ids, display_ids_int))
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    key = row['DisplayID'].strip()
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for DisplayID {key}: {row['Capacity']}")
    capacity_dict[key] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName'].strip()
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wt
I = display_ids
J = product_names
for i in I:
    if i not in capacity_dict:
        raise KeyError(f'Missing capacity for display area {i}')
for j in J:
    if j not in value_dict or j not in weight_dict:
        raise KeyError(f'Missing value or weight for product {j}')
m = gp.Model('BoatDisplayAllocation')
x_vars = m.addVars(I, J, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in I for j in J)), gp.GRB.MAXIMIZE)
for i in I:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in J)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()