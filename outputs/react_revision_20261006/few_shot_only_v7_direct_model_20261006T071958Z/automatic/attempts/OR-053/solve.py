import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with supported encodings.')
    shelves_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv'
    shelves_df = read_csv_robust(shelves_path)
    products_df = read_csv_robust(products_path)
    I = shelves_df['ShelfID'].tolist()
    J = products_df['ProductName'].tolist()
    try:
        c_i = shelves_df.set_index('ShelfID')['Capacity'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error processing shelf capacities: {e}')
    try:
        v_j = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error processing product values: {e}')
    try:
        w_j = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
    except Exception as e:
        raise RuntimeError(f'Error processing product weights: {e}')
    missing_c = [i for i in I if i not in c_i]
    missing_v = [j for j in J if j not in v_j]
    missing_w = [j for j in J if j not in w_j]
    if missing_c or missing_v or missing_w:
        raise ValueError(f'Missing data: Shelf capacities missing for {missing_c}, product values missing for {missing_v}, product weights missing for {missing_w}')
    m = gp.Model('BigMart_Shelf_Allocation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i])
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in quantity_vars.values():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()