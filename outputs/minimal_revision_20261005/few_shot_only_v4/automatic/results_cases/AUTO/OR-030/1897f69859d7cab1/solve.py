import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    mask = df['Product Name'].astype(str).str.casefold() == 'fdk57'
    df_fdk57 = df[mask].copy()
    if df_fdk57.empty:
        raise ValueError("No rows found for Product Name classified under 'FDK57'.")
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_fdk57.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_fdk57.index.tolist()
    try:
        revenue = df_fdk57['Revenue'].to_dict()
        demand = df_fdk57['Demand'].to_dict()
        inventory = df_fdk57['Initial Inventory'].to_dict()
    except Exception as e:
        raise ValueError(f'Error extracting parameters: {e}')
    for i in items:
        for (param, dic) in [('Revenue', revenue), ('Demand', demand), ('Initial Inventory', inventory)]:
            if i not in dic:
                raise ValueError(f'Missing {param} for item {i}')
            if pd.isnull(dic[i]):
                raise ValueError(f'Null {param} for item {i}')
            try:
                float(dic[i])
            except Exception:
                raise ValueError(f'Non-numeric {param} for item {i}: {dic[i]}')
    revenue = {i: float(revenue[i]) for i in items}
    demand = {i: int(demand[i]) for i in items}
    inventory = {i: int(inventory[i]) for i in items}
    m = gp.Model('FDK57_Car_Dealership')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()