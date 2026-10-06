import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'
    for enc in csv_encodings:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with tried encodings.')
    for enc in csv_encodings:
        try:
            products_df = pd.read_csv(products_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with tried encodings.')
    if not {'VehicleID', 'VehicleType', 'Capacity'}.issubset(capacity_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
        raise ValueError('products.csv missing required columns.')
    capacity_df['VehicleType_cf'] = capacity_df['VehicleType'].astype(str).str.casefold()
    products_df['ProductName_cf'] = products_df['ProductName'].astype(str).str.casefold()
    merged = pd.merge(capacity_df, products_df, left_on='VehicleType_cf', right_on='ProductName_cf', how='inner', suffixes=('_cap', '_prod'))
    if merged.empty:
        raise ValueError('No matching vehicle types between capacity.csv and products.csv.')
    vehicle_keys = []
    b = {}
    u = {}
    vehicle_info = {}
    for (_, row) in merged.iterrows():
        key = row['VehicleID']
        if key in b:
            u[key] += row['Capacity']
            if b[key] != row['Value']:
                raise ValueError(f'Conflicting Value for VehicleID {key}')
        else:
            vehicle_keys.append(key)
            b[key] = row['Value']
            u[key] = row['Capacity']
            vehicle_info[key] = {'VehicleType': row['VehicleType'], 'ProductName': row['ProductName'], 'Weight': row['Weight']}
    if not len(vehicle_keys) == len(b) == len(u):
        raise ValueError('Mismatch in vehicle key, benefit, or capacity dimensions.')
    m = gp.Model('NewCarSalesInNorway1')
    x = m.addVars(vehicle_keys, lb=0, ub=[u[k] for k in vehicle_keys], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b[k] * x[k] for k in vehicle_keys)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[k] for k in vehicle_keys)) <= sum((u[k] for k in vehicle_keys)), name='total_capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()