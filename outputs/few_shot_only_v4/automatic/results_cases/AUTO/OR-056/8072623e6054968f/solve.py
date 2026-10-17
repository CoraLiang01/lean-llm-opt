import gurobipy as gp
from gurobipy import GRB
import pandas as pd
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv', dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must contain DisplayID and Capacity columns')
display_ids = list(capacity_df['DisplayID'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['DisplayID']] = int(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity value for DisplayID {row['DisplayID']}")
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must contain ProductName, Value, and Weight columns')
product_names = list(products_df['ProductName'])
value = {}
weight = {}
for idx, row in products_df.iterrows():
    try:
        value[row['ProductName']] = int(row['Value'])
        weight[row['ProductName']] = int(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {row['ProductName']}")
if len(display_ids) == 0 or len(product_names) == 0:
    raise ValueError('No display areas or products found in the input files')
for did in display_ids:
    if did not in capacity:
        raise ValueError(f'Missing capacity for DisplayID {did}')
for pname in product_names:
    if pname not in value or pname not in weight:
        raise ValueError(f'Missing value or weight for ProductName {pname}')
m = gp.Model('Boat_Display_Optimization')
x_vars = m.addVars(display_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[pname] * x_vars[did, pname] for did in display_ids for pname in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[pname] * x_vars[did, pname] for pname in product_names)) <= capacity[did] for did in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')