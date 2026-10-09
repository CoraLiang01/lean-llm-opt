import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    cabinets_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
    cabinets_df = read_csv_robust(cabinets_path, dtype=str, keep_default_na=False)
    products_df = read_csv_robust(products_path, dtype=str, keep_default_na=False)
    I = cabinets_df['CabinetID'].tolist()
    J = products_df['ProductName'].tolist()
    if cabinets_df['CabinetID'].duplicated().any():
        raise ValueError('Duplicate CabinetID found in capacity.csv')
    c_i = {}
    for (_, row) in cabinets_df.iterrows():
        try:
            c_i[row['CabinetID']] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for CabinetID {row['CabinetID']}: {row['Capacity']}")
    if products_df['ProductName'].duplicated().any():
        raise ValueError('Duplicate ProductName found in products.csv')
    v_j = {}
    w_j = {}
    for (_, row) in products_df.iterrows():
        try:
            v_j[row['ProductName']] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {row['ProductName']}: {row['Value']}")
        try:
            w_j[row['ProductName']] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {row['ProductName']}: {row['Weight']}")
    if set(I) != set(c_i.keys()):
        raise ValueError('Mismatch between cabinets index set and c_i keys')
    if set(J) != set(v_j.keys()) or set(J) != set(w_j.keys()):
        raise ValueError('Mismatch between products index set and v_j/w_j keys')
    m = gp.Model('CoffeeCabinetAllocation')
    quantity_vars = m.addVars(I, J, lb=0, vtype=GRB.INTEGER, name='')
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