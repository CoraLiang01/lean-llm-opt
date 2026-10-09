import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'capacity': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'products': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_df = read_csv_with_encodings(csv_paths['capacity'])
    products_df = read_csv_with_encodings(csv_paths['products'])
    for col in ['VehicleID', 'VehicleType', 'Capacity']:
        if col not in capacity_df.columns:
            raise ValueError(f'Missing column {col} in capacity.csv')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f'Missing column {col} in products.csv')
    capacity_df['Capacity'] = pd.to_numeric(capacity_df['Capacity'], errors='raise')
    vehicle_capacity = capacity_df.groupby('VehicleType')['Capacity'].sum().to_dict()
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    vehicle_types = set(vehicle_capacity.keys())
    product_names = set(products_df['ProductName'])
    matched_types = vehicle_types & product_names
    if not matched_types:
        raise ValueError('No matching VehicleType/ProductName between capacity.csv and products.csv')
    vehicle_types = sorted(matched_types)
    benefit = products_df.set_index('ProductName').loc[vehicle_types, 'Value'].to_dict()
    capacity = {vt: vehicle_capacity[vt] for vt in vehicle_types}
    m = gp.Model('NewCarSalesInNorway1')
    quantity_vars = m.addVars(vehicle_types, lb=0, ub=[capacity[vt] for vt in vehicle_types], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((benefit[vt] * quantity_vars[vt] for vt in vehicle_types)), GRB.MAXIMIZE)
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