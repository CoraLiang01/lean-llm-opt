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
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
    products_df = read_csv_with_encodings(products_path)
    capacity_df = read_csv_with_encodings(capacity_path)
    for col in ['ProductName', 'Value', 'Weight']:
        if col not in products_df.columns:
            raise ValueError(f"Missing column '{col}' in products.csv")
    if 'Capacity' not in capacity_df.columns:
        raise ValueError("Missing column 'Capacity' in capacity.csv")
    I = products_df['ProductName'].tolist()
    products_grouped = products_df.groupby('ProductName', sort=False, dropna=False).agg({'Value': lambda x: list(x), 'Weight': lambda x: list(x)}).reset_index()
    p_i = {}
    w_i = {}
    for (idx, row) in products_grouped.iterrows():
        pname = row['ProductName']
        values = row['Value']
        weights = row['Weight']
        if len(values) != 1 or len(weights) != 1:
            raise ValueError(f"Multiple Value or Weight entries for ProductName '{pname}'")
        try:
            p_i[pname] = float(values[0])
        except Exception:
            raise ValueError(f"Non-numeric Value for ProductName '{pname}': {values[0]}")
        try:
            w_i[pname] = float(weights[0])
        except Exception:
            raise ValueError(f"Non-numeric Weight for ProductName '{pname}': {weights[0]}")
    I = list(p_i.keys())
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must have exactly one row for total inventory capacity')
    try:
        C = float(capacity_df.iloc[0]['Capacity'])
    except Exception:
        raise ValueError(f"Non-numeric Capacity: {capacity_df.iloc[0]['Capacity']}")
    m = gp.Model('CarSales_Inventory')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((p_i[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_i[i] * quantity_vars[i] for i in I)) <= C, name='capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()