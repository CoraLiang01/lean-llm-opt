import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path):
        encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f'Could not decode {path} with tried encodings.')
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA14/AmazonProductsSalesDataset2023/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'StorageID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    I = cap_df['StorageID'].astype(str).unique().tolist()
    J = prod_df['ProductName'].astype(str).unique().tolist()
    c_i = {}
    for (_, row) in cap_df.iterrows():
        key = str(row['StorageID'])
        val = row['Capacity']
        if pd.isnull(val):
            raise ValueError(f'Missing Capacity for StorageID {key}')
        if key in c_i:
            c_i[key] += val
        else:
            c_i[key] = val
    v_j = {}
    w_j = {}
    for (_, row) in prod_df.iterrows():
        key = str(row['ProductName'])
        vval = row['Value']
        wval = row['Weight']
        if pd.isnull(vval) or pd.isnull(wval):
            raise ValueError(f'Missing Value or Weight for ProductName {key}')
        if key in v_j:
            v_j[key] += vval
            w_j[key] += wval
        else:
            v_j[key] = vval
            w_j[key] = wval
    for i in I:
        if i not in c_i:
            raise ValueError(f'Missing capacity for StorageID {i}')
    for j in J:
        if j not in v_j or j not in w_j:
            raise ValueError(f'Missing value/weight for ProductName {j}')
    m = gp.Model('Amazon_AC_Storage')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_j[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(gp.quicksum((w_j[j] * x[i, j] for j in J)) <= c_i[i], name=f'cap_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()