import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    mask = df['Product Name'].str.casefold().str.contains('baby')
    baby_df = df[mask].copy()
    items = baby_df['Product Name'].tolist()
    try:
        revenue = {}
        demand = {}
        inventory = {}
        for (idx, row) in baby_df.iterrows():
            key = row['Product Name']
            try:
                A_i = float(row['Revenue'])
                d_i = float(row['Demand'])
                I_i = float(row['Initial Inventory'])
            except Exception:
                raise ValueError(f"Non-numeric or missing parameter for product '{key}'")
            if key in revenue:
                revenue[key] += A_i
                demand[key] += d_i
                inventory[key] += I_i
            else:
                revenue[key] = A_i
                demand[key] = d_i
                inventory[key] = I_i
    except KeyError as e:
        raise ValueError(f'Missing required column: {e}')
    for key in items:
        if key not in revenue or key not in demand or key not in inventory:
            raise ValueError(f"Missing parameter for product '{key}'")
    m = gp.Model('Baby_Product_Revenue_Maximization')
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()