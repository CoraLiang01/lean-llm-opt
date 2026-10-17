import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'DisplayID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must contain DisplayID and Capacity columns')
display_ids = capacity_df['DisplayID'].tolist()
capacity_dict = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity_dict[row['DisplayID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for DisplayID {row['DisplayID']}")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must contain ProductName, Value, and Weight columns')
product_names = products_df['ProductName'].tolist()
value_dict = {}
weight_dict = {}
for idx, row in products_df.iterrows():
    try:
        value_dict[row['ProductName']] = float(row['Value'])
        weight_dict[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid value or weight for ProductName {row['ProductName']}")
if len(display_ids) == 0 or len(product_names) == 0:
    raise ValueError('No display areas or products found in the CSV files')
for did in display_ids:
    if did not in capacity_dict:
        raise ValueError(f'Missing capacity for DisplayID {did}')
for pname in product_names:
    if pname not in value_dict or pname not in weight_dict:
        raise ValueError(f'Missing value or weight for ProductName {pname}')
m = gp.Model('Boat_Display_Allocation')
x_vars = m.addVars(display_ids, product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value_dict[pname] * x_vars[did, pname] for did in display_ids for pname in product_names)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight_dict[pname] * x_vars[did, pname] for pname in product_names)) <= capacity_dict[did] for did in display_ids), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')