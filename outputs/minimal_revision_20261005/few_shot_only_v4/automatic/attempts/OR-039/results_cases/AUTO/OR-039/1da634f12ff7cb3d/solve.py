import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            products_df = pd.read_csv(products_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {products_path} with supported encodings.')
    for encoding in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
        try:
            capacity_df = pd.read_csv(capacity_path, encoding=encoding)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {capacity_path} with supported encodings.')
    if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
        raise ValueError('products.csv must contain columns: ProductName, Value, Weight')
    if 'Warehouse ID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
        raise ValueError('capacity.csv must contain columns: Warehouse ID, Capacity')
    I = products_df['ProductName'].astype(str).unique().tolist()
    J = capacity_df['Warehouse ID'].astype(str).unique().tolist()
    v_i = products_df.set_index('ProductName')['Value'].to_dict()
    w_i = products_df.set_index('ProductName')['Weight'].to_dict()
    C_j = capacity_df.set_index('Warehouse ID')['Capacity'].to_dict()
    for i in I:
        if i not in v_i or i not in w_i:
            raise ValueError(f'Missing value or weight for product {i}')
    for j in J:
        if j not in C_j:
            raise ValueError(f'Missing capacity for warehouse {j}')
    m = gp.Model('Car_Dealership_Inventory')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_i[i] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_i[i] * x[i, j] for i in I)) <= C_j[j] for j in J), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()