import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = {'capacity': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/capacity.csv', 'products': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA4/NewCarSalesInNorway1/products.csv'}
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    df_capacity = read_csv(csv_paths['capacity'])
    df_products = read_csv(csv_paths['products'])
    df_capacity['VehicleType_norm'] = df_capacity['VehicleType'].astype(str).str.casefold()
    df_products['ProductName_norm'] = df_products['ProductName'].astype(str).str.casefold()
    df_join = pd.merge(df_capacity, df_products, left_on='VehicleType_norm', right_on='ProductName_norm', how='inner', suffixes=('_cap', '_prod'))
    for col in ['VehicleType', 'Capacity', 'ProductName', 'Value']:
        if col not in df_join.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_join['VehicleType'])
    b_i = {}
    u_i = {}
    for (_, row) in df_join.iterrows():
        vt = row['VehicleType']
        if vt in b_i:
            if b_i[vt] != row['Value']:
                raise ValueError(f'Conflicting Value for VehicleType {vt}')
            u_i[vt] += row['Capacity']
        else:
            b_i[vt] = row['Value']
            u_i[vt] = row['Capacity']
    if not I or not b_i or (not u_i):
        raise ValueError('No valid vehicle types with both capacity and value found.')
    m = gp.Model('NewCarSalesInNorway1')
    m.Params.MIPGap = 0.0001
    x = m.addVars(I, lb=0, ub=[u_i[i] for i in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= u_i[i] for i in I), name='')
    total_capacity = sum((u_i[i] for i in I))
    m.addConstr(gp.quicksum((x[i] for i in I)) <= total_capacity, name='total_capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()