import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
sections = [row['SectionID'] for _, row in capacity_df.iterrows()]
products = [row['ProductName'] for _, row in products_df.iterrows()]
capacity = {}
for _, row in capacity_df.iterrows():
    try:
        capacity[row['SectionID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for SectionID {row['SectionID']}")
value = {}
weight = {}
for _, row in products_df.iterrows():
    try:
        value[row['ProductName']] = float(row['Value'])
        weight[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {row['ProductName']}")
if set(capacity.keys()) != set(sections):
    raise ValueError('Mismatch in section identifiers between capacity data and index set.')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between products data and index set.')
m = gp.Model('Supermarket_Section_Allocation')
x_vars = m.addVars(sections, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in sections for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in sections), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')