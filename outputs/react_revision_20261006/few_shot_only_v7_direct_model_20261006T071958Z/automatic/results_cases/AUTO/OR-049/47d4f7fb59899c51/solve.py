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
    shelves_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
    shelves_df = read_csv_with_encodings(shelves_path)
    products_df = read_csv_with_encodings(products_path)
    I = list(shelves_df['ShelfID'])
    J = list(products_df['ProductName'])
    if not set(['ShelfID', 'Capacity']).issubset(shelves_df.columns):
        raise ValueError('Missing required columns in shelves data.')
    c_i = {}
    for (_, row) in shelves_df.iterrows():
        shelf = row['ShelfID']
        try:
            cap = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf}: {row['Capacity']}")
        c_i[shelf] = cap
    if not set(['ProductName', 'Value', 'Weight']).issubset(products_df.columns):
        raise ValueError('Missing required columns in products data.')
    v_j = {}
    w_j = {}
    for (_, row) in products_df.iterrows():
        prod = row['ProductName']
        try:
            val = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {prod}: {row['Value']}")
        try:
            wt = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {prod}: {row['Weight']}")
        v_j[prod] = val
        w_j[prod] = wt
    if set(I) != set(c_i.keys()):
        raise ValueError('Mismatch between shelf index set and capacity keys.')
    if set(J) != set(v_j.keys()) or set(J) != set(w_j.keys()):
        raise ValueError('Mismatch between product index set and value/weight keys.')
    m = gp.Model('Retail_Shelf_Allocation')
    m.Params.MIPGap = 0.0001
    quantity_keys = [(i, j) for i in I for j in J]
    quantity_vars = m.addVars(quantity_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * quantity_vars[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * quantity_vars[i, j] for j in J)) <= c_i[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()