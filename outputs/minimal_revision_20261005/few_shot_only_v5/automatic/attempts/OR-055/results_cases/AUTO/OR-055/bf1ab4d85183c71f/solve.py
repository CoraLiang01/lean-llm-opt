import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv'
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
    if cap['DisplayID'].duplicated().any():
        cap = cap.groupby('DisplayID', as_index=False).agg({'Capacity': 'sum'})
    display_ids = cap['DisplayID'].tolist()
    capacity_dict = dict(zip(cap['DisplayID'], cap['Capacity']))
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv'
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
    if prod['ProductName'].duplicated().any():
        prod = prod.groupby('ProductName', as_index=False).agg({'Value': 'first', 'Weight': 'first'})
    product_names = prod['ProductName'].tolist()
    value_dict = dict(zip(prod['ProductName'], prod['Value']))
    weight_dict = dict(zip(prod['ProductName'], prod['Weight']))
    for d in [capacity_dict]:
        if any((pd.isnull(v) for v in d.values())):
            raise ValueError('Missing capacity values')
    for d in [value_dict, weight_dict]:
        if any((pd.isnull(v) for v in d.values())):
            raise ValueError('Missing product value/weight')
    I = display_ids
    J = product_names
    m = gp.Model('Boat_Display_Allocation')
    x_keys = [(i, j) for i in I for j in J]
    x = m.addVars(x_keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((weight_dict[j] * x[i, j] for j in J)) <= capacity_dict[i] for i in I), name='')
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