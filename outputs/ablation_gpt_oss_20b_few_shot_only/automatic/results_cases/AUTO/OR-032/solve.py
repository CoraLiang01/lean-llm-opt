import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df[col] = df[col].astype(float)
    books_df = df[df['Product_Name'].str.startswith('Books_')].copy()
    rev = books_df.set_index('Product_Name')['Revenue'].to_dict()
    dem = books_df.set_index('Product_Name')['Demand'].to_dict()
    inv = books_df.set_index('Product_Name')['Initial Inventory'].to_dict()
    keys_rev = set(rev.keys())
    keys_dem = set(dem.keys())
    keys_inv = set(inv.keys())
    if keys_rev != keys_dem or keys_rev != keys_inv:
        missing_rev = keys_dem | keys_inv - keys_rev
        missing_dem = keys_rev | keys_inv - keys_dem
        missing_inv = keys_rev | keys_dem - keys_inv
        raise ValueError(f'Missing data for keys: rev={missing_rev}, dem={missing_dem}, inv={missing_inv}')
    model = gurobipy.Model('BookSales')
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(rev.keys(), lb=0, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((rev[i] * x[i] for i in rev)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= dem[i] for i in rev), name='')
    model.addConstrs((x[i] <= inv[i] for i in rev), name='')
    model.optimize()
    return model
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')