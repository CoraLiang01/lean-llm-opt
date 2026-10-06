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
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv'
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv'
    cap_df = read_csv_with_encodings(capacity_path)
    prod_df = read_csv_with_encodings(products_path)
    if not {'SectionID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    sections = cap_df['SectionID'].astype(str).unique().tolist()
    products = prod_df['ProductName'].astype(str).unique().tolist()
    cap_dict = {}
    for (_, row) in cap_df.iterrows():
        sid = str(row['SectionID'])
        if sid in cap_dict:
            cap_dict[sid] += row['Capacity']
        else:
            cap_dict[sid] = row['Capacity']
    value_dict = {}
    weight_dict = {}
    for (_, row) in prod_df.iterrows():
        pname = str(row['ProductName'])
        if pname in value_dict:
            value_dict[pname] += row['Value']
            weight_dict[pname] += row['Weight']
        else:
            value_dict[pname] = row['Value']
            weight_dict[pname] = row['Weight']
    for s in sections:
        if s not in cap_dict:
            raise ValueError(f'Section {s} missing capacity.')
    for p in products:
        if p not in value_dict or p not in weight_dict:
            raise ValueError(f'Product {p} missing value or weight.')
    keys = [(s, p) for s in sections for p in products]
    m = gp.Model('Supermarket_Stocking')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[p] * x[s, p] for s in sections for p in products)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight_dict[p] * x[s, p] for p in products)) <= cap_dict[s] for s in sections), name='')
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