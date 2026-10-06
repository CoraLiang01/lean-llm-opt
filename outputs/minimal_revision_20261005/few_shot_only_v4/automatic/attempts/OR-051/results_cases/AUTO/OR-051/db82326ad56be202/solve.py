import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cap_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
    prod_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cap_df = pd.read_csv(cap_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {cap_path} with tried encodings.')
    for enc in encodings:
        try:
            prod_df = pd.read_csv(prod_path, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read {prod_path} with tried encodings.')
    if not {'CabinetID', 'Capacity'}.issubset(cap_df.columns):
        raise ValueError('capacity.csv missing required columns.')
    if not {'ProductName', 'Value', 'Weight'}.issubset(prod_df.columns):
        raise ValueError('products.csv missing required columns.')
    C = cap_df['CabinetID'].astype(str).unique().tolist()
    P = prod_df['ProductName'].astype(str).unique().tolist()
    cap_dict = {}
    for (_, row) in cap_df.iterrows():
        key = str(row['CabinetID'])
        val = row['Capacity']
        if key in cap_dict:
            cap_dict[key] += val
        else:
            cap_dict[key] = val
    v_dict = {}
    w_dict = {}
    for (_, row) in prod_df.iterrows():
        key = str(row['ProductName'])
        v = row['Value']
        w = row['Weight']
        if key in v_dict:
            v_dict[key] += v
            w_dict[key] += w
        else:
            v_dict[key] = v
            w_dict[key] = w
    for i in C:
        if i not in cap_dict:
            raise ValueError(f'Missing capacity for cabinet {i}')
    for j in P:
        if j not in v_dict or j not in w_dict:
            raise ValueError(f'Missing value/weight for product {j}')
    keys = [(i, j) for i in C for j in P]
    m = gp.Model('CoffeeCabinetAllocation')
    x = m.addVars(keys, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_dict[j] * x[i, j] for i in C for j in P)), GRB.MAXIMIZE)
    m.addConstrs((gp.quicksum((w_dict[j] * x[i, j] for j in P)) <= cap_dict[i] for i in C), name='')
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