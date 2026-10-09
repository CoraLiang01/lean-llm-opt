import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
    for enc in csv_encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {capacity_path} with tried encodings.')
    for enc in csv_encodings:
        try:
            products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Failed to read {products_path} with tried encodings.')
    if not {'DisplayID', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    display_ids = capacity_df['DisplayID'].tolist()
    product_names = products_df['ProductName'].tolist()
    try:
        capacity_dict = dict(zip(capacity_df['DisplayID'], capacity_df['Capacity'].astype(float)))
    except Exception:
        raise ValueError('Non-numeric or missing Capacity in capacity.csv.')
    try:
        value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(float)))
        weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(float)))
    except Exception:
        raise ValueError('Non-numeric or missing Value/Weight in products.csv.')
    for i in display_ids:
        if i not in capacity_dict:
            raise ValueError(f'Missing capacity for display area {i}')
    for j in product_names:
        if j not in value_dict or j not in weight_dict:
            raise ValueError(f'Missing value/weight for product {j}')
    keys = [(i, j) for i in display_ids for j in product_names]
    m = gp.Model('Boat_Display_Optimization')
    quantity_vars = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[j] * quantity_vars[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight_dict[j] * quantity_vars[i, j] for j in product_names)) <= capacity_dict[i] for i in display_ids), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')