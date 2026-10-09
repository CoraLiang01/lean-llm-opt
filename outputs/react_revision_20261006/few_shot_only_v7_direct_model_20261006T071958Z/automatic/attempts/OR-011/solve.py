import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError('Failed to read CSV with supported encodings.')
    df_id999 = df[df['id_number'].str.casefold() == 'id999']
    required_cols = ['id_number', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_id999.columns:
            raise ValueError(f'Missing required column: {col}')
    I = list(df_id999.index)
    try:
        revenue = df_id999['Revenue'].astype(float).to_dict()
        demand = df_id999['Demand'].astype(float).to_dict()
        inventory = df_id999['Initial Inventory'].astype(float).to_dict()
    except Exception as e:
        raise ValueError(f'Error converting parameters to float: {e}')
    for i in I:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for index {i}')
    m = gp.Model('Supermarket_id999_Fulfillment')
    quantity_vars = m.addVars(I, lb=0, ub=[min(inventory[i], demand[i]) for i in I], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in I), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in I), name='')
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