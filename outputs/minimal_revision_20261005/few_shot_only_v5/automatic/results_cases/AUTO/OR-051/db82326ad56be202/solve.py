import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
    try:
        cap_df = pd.read_csv(capacity_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            cap_df = pd.read_csv(capacity_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                cap_df = pd.read_csv(capacity_path, encoding='gbk')
            except UnicodeDecodeError:
                cap_df = pd.read_csv(capacity_path, encoding='latin-1')
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
    try:
        prod_df = pd.read_csv(products_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            prod_df = pd.read_csv(products_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                prod_df = pd.read_csv(products_path, encoding='gbk')
            except UnicodeDecodeError:
                prod_df = pd.read_csv(products_path, encoding='latin-1')
    if not {'CabinetID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns')
    cabinets = cap_df['CabinetID'].astype(str).unique().tolist()
    products = prod_df['ProductName'].astype(str).unique().tolist()
    cap_dict = {}
    for (_, row) in cap_df.iterrows():
        cab = str(row['CabinetID'])
        cap = row['Capacity']
        if cab in cap_dict:
            cap_dict[cab] += cap
        else:
            cap_dict[cab] = cap
    value_dict = {}
    weight_dict = {}
    for (_, row) in prod_df.iterrows():
        prod = str(row['ProductName'])
        val = row['Value']
        wt = row['Weight']
        if prod in value_dict:
            value_dict[prod] += val
            weight_dict[prod] += wt
        else:
            value_dict[prod] = val
            weight_dict[prod] = wt
    for cab in cabinets:
        if cab not in cap_dict:
            raise ValueError(f'Missing capacity for cabinet {cab}')
    for prod in products:
        if prod not in value_dict or prod not in weight_dict:
            raise ValueError(f'Missing value/weight for product {prod}')
    keys = [(cab, prod) for cab in cabinets for prod in products]
    m = gp.Model('CoffeeCabinetAllocation')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((value_dict[prod] * x[cab, prod] for cab in cabinets for prod in products)), GRB.MAXIMIZE)
    for cab in cabinets:
        m.addConstr(gp.quicksum((weight_dict[prod] * x[cab, prod] for prod in products)) <= cap_dict[cab], name=f'cap_{cab}')
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