import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
    try:
        cap = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            cap = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cap = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                cap = pd.read_csv(capacity_path, encoding='latin-1')
    if not {'DisplayID', 'Capacity'}.issubset(cap.columns):
        raise ValueError('Missing required columns in capacity.csv')
    cap['DisplayID'] = cap['DisplayID'].astype(str)
    display_ids = cap['DisplayID'].unique().tolist()
    capacity_dict = cap.groupby('DisplayID')['Capacity'].sum().to_dict()
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
    try:
        prod = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            prod = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                prod = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                prod = pd.read_csv(products_path, encoding='latin-1')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod.columns):
        raise ValueError('Missing required columns in products.csv')
    prod['ProductName'] = prod['ProductName'].astype(str)
    product_names = prod['ProductName'].unique().tolist()
    value_dict = prod.groupby('ProductName')['Value'].sum().to_dict()
    weight_dict = prod.groupby('ProductName')['Weight'].sum().to_dict()
    for i in display_ids:
        if i not in capacity_dict:
            raise ValueError(f'Missing capacity for display area {i}')
    for j in product_names:
        if j not in value_dict or j not in weight_dict:
            raise ValueError(f'Missing value or weight for product {j}')
    keys = [(i, j) for i in display_ids for j in product_names]
    m = gp.Model('BoatDealershipDisplay')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in display_ids for j in product_names)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight_dict[j] * x[i, j] for j in product_names)) <= capacity_dict[i] for i in display_ids), name='')
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