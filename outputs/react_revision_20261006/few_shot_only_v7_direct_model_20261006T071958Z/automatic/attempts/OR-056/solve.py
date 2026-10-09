import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in csv_encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    I = capacity_df['DisplayID'].tolist()
    J = products_df['ProductName'].tolist()
    try:
        c_i = capacity_df.set_index('DisplayID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError('Error converting Capacity to float in capacity.csv') from e
    try:
        v_j = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError('Error converting Value to float in products.csv') from e
    try:
        w_j = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError('Error converting Weight to float in products.csv') from e
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for DisplayID {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value or weight for ProductName {j}')
    m = gp.Model('Boat_Display_Assignment')
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i] for i in I), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()