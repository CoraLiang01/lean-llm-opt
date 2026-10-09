import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_paths = ['/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv', '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv']
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_fallback(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    products_df = read_csv_with_fallback(csv_paths[0])
    capacity_df = read_csv_with_fallback(csv_paths[1])
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing column 'Capacity' in capacity.csv")
    products_df = products_df.copy()
    products_df['Value'] = pd.to_numeric(products_df['Value'], errors='raise')
    products_df['Weight'] = pd.to_numeric(products_df['Weight'], errors='raise')
    I = products_df['ProductName'].tolist()
    value = dict(zip(products_df['ProductName'], products_df['Value']))
    weight = dict(zip(products_df['ProductName'], products_df['Weight']))
    capacity_vals = pd.to_numeric(capacity_df['Capacity'], errors='raise')
    C = capacity_vals.sum()
    for i in I:
        if i not in value or i not in weight:
            raise ValueError(f"Missing value or weight for product '{i}'")
    m = gp.Model('NYC_RealEstate_Development')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()