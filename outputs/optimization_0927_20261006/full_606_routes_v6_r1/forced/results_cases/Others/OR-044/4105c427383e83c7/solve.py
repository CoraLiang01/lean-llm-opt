import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'SectionID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'SectionID' and 'Capacity' columns.")
capacity_df['SectionID'] = capacity_df['SectionID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
sections = capacity_df['SectionID'].tolist()
section_capacities = {}
for (idx, row) in capacity_df.iterrows():
    sid = row['SectionID']
    try:
        cap = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for SectionID {sid}: {row['Capacity']}")
    section_capacities[sid] = cap
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv', sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
products = products_df['ProductName'].tolist()
product_values = {}
product_weights = {}
for (idx, row) in products_df.iterrows():
    pid = row['ProductName']
    try:
        val = int(row['Value'])
        wt = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {pid}: Value={row['Value']}, Weight={row['Weight']}")
    product_values[pid] = val
    product_weights[pid] = wt
if set(section_capacities.keys()) != set(sections):
    raise ValueError('Mismatch in section IDs between capacity data and section list.')
if set(product_values.keys()) != set(products) or set(product_weights.keys()) != set(products):
    raise ValueError('Mismatch in product IDs between product data and product list.')
m = gp.Model('SupermarketProductSelection')
x_vars = m.addVars(sections, products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[pid] * x_vars[sid, pid] for sid in sections for pid in products)), gp.GRB.MAXIMIZE)
for sid in sections:
    m.addConstr(gp.quicksum((product_weights[pid] * x_vars[sid, pid] for pid in products)) <= section_capacities[sid], name=f'Capacity_{sid}')
m.optimize()