import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'file_0_view_0': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'file_1_view_0': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'}

    def read_csv(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    df_capacity = read_csv(csv_paths['file_0_view_0'])
    df_products = read_csv(csv_paths['file_1_view_0'])
    I = df_products['ProductName'].unique().tolist()
    b_i = {}
    w_i = {}
    for (_, row) in df_products.iterrows():
        key = row['ProductName']
        if key in b_i:
            raise ValueError(f'Duplicate ProductName in products.csv: {key}')
        try:
            b_i[key] = float(row['Value'])
        except Exception:
            raise ValueError(f"Non-numeric Value for ProductName {key}: {row['Value']}")
        try:
            w_i[key] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Non-numeric Weight for ProductName {key}: {row['Weight']}")
    cap_map = {}
    for (_, row) in df_capacity.iterrows():
        vt = row['VehicleType']
        if vt in cap_map:
            raise ValueError(f'Duplicate VehicleType in capacity.csv: {vt}')
        try:
            cap_map[vt] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Non-numeric Capacity for VehicleType {vt}: {row['Capacity']}")
    c_i = {}
    for i in I:
        if i not in cap_map:
            raise ValueError(f'Missing capacity for ProductName/VehicleType {i}')
        c_i[i] = cap_map[i]
    m = gp.Model('NewCarSalesInNorway')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, ub=[c_i[i] for i in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()