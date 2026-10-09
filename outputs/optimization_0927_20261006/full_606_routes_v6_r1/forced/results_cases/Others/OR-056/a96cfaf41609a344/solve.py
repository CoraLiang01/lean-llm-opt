import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if not {'DisplayID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain 'DisplayID' and 'Capacity' columns.")
capacity_df['DisplayID'] = capacity_df['DisplayID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
display_ids = list(capacity_df['DisplayID'])
display_id_set = set(display_ids)
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    did = row['DisplayID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for DisplayID {did}: {row['Capacity']}")
    capacity_dict[did] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv', sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = list(products_df['ProductName'])
product_name_set = set(product_names)
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pname}: Value={row['Value']}, Weight={row['Weight']}")
    value_dict[pname] = val
    weight_dict[pname] = wt
if len(display_ids) == 0 or len(product_names) == 0:
    raise ValueError('No display areas or product types found in the input files.')
m = gp.Model('BoatDisplayAssignment')
x_vars = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in display_ids for j in product_names)), gp.GRB.MAXIMIZE)
for i in display_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_names)) <= capacity_dict[i], name=f'capacity_{i}')
m.optimize()