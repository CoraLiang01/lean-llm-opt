import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    I = products_df['ProductName'].tolist()
    J = capacity_df['Warehouse ID'].tolist()
    try:
        v_i = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
        w_i = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value/Weight to float: {e}')
    try:
        C_j = capacity_df.set_index('Warehouse ID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    missing_v = [i for i in I if i not in v_i]
    missing_w = [i for i in I if i not in w_i]
    missing_C = [j for j in J if j not in C_j]
    if missing_v or missing_w or missing_C:
        raise ValueError(f'Missing parameter values: v_i missing for {missing_v}, w_i missing for {missing_w}, C_j missing for {missing_C}')
    m = gp.Model('NewCarSalesInNorway2')
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_i[i] * quantity_vars[i, j] for i in I)) <= C_j[j] for j in J), name='')
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