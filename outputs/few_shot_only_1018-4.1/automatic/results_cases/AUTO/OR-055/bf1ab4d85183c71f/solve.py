import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['DisplayID'] = capacity_df['DisplayID'].astype(str).str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
display_ids = capacity_df['DisplayID'].tolist()
capacity = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv', dtype=str, keep_default_na=False, sep='\\s{2,}|\\t', engine='python')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
product_names = products_df['ProductName'].tolist()
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(capacity.keys()) != set(display_ids):
    raise ValueError('Mismatch in display area identifiers.')
if set(value.keys()) != set(product_names) or set(weight.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers.')
m = gp.Model('Boat_Display_Allocation')
x_vars = m.addVars(display_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in product_names)) <= capacity[i] for i in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')