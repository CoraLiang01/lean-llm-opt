import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        last_err = None
        for enc in csv_encodings:
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
            except Exception as e:
                last_err = e
        raise last_err
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA7/NYCPropertySales2/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row for total capacity')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError('Capacity value in capacity.csv is not a valid number')
    I = []
    v = {}
    w = {}
    for (idx, row) in products_df.iterrows():
        i = row['ProductName']
        if i in v:
            raise ValueError(f"Duplicate ProductName '{i}' in products.csv")
        try:
            v_i = float(row['Value'])
            w_i = float(row['Weight'])
        except Exception:
            raise ValueError(f"Non-numeric Value or Weight for ProductName '{i}'")
        I.append(i)
        v[i] = v_i
        w[i] = w_i
    m = gp.Model('NYC_RealEstate_Development')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')