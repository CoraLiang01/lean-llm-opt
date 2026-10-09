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
    shelves_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
    shelves_df = read_csv_with_encodings(shelves_path)
    products_df = read_csv_with_encodings(products_path)
    for col in ['ShelfID', 'Capacity']:
        if col not in shelves_df.columns:
            raise ValueError(f'Missing column {col} in {shelves_path}')
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f'Missing column {col} in {products_path}')
    I = list(shelves_df['ShelfID'])
    J = list(products_df['ProductName'])
    try:
        c_i = shelves_df.set_index('ShelfID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Capacity to float: {e}')
    try:
        v_j = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
        w_j = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting Value/Weight to float: {e}')
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for shelf {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value or weight for product {j}')
    m = gp.Model('BigMart_Shelf_Allocation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i])
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')