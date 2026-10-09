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
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv'
    capacity_df = read_csv_with_encodings(capacity_path)
    products_df = read_csv_with_encodings(products_path)
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing 'Capacity' column in capacity.csv")
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Missing '{col}' column in products.csv")
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one row for total capacity.')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError('Capacity value in capacity.csv is not a valid number.')
    I = []
    b = {}
    w = {}
    for (idx, row) in products_df.iterrows():
        prod = row['ProductName']
        if prod in I:
            raise ValueError(f"Duplicate ProductName '{prod}' in products.csv")
        try:
            b[prod] = float(row['Value'])
            w[prod] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Non-numeric Value or Weight for product '{prod}'")
        I.append(prod)
    if set(b.keys()) != set(I) or set(w.keys()) != set(I):
        raise ValueError('Mismatch in product keys and coefficients.')
    m = gp.Model('Pharmacy_Inventory')
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((b[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in quantity_vars.values():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()