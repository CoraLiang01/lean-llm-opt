import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
    I = df['Product Name'].tolist()
    r_i = dict(zip(df['Product Name'], df['Revenue']))
    d_i = dict(zip(df['Product Name'], df['Demand']))
    I_i = dict(zip(df['Product Name'], df['Initial Inventory']))
    for key in I:
        if key not in r_i or key not in d_i or key not in I_i:
            raise ValueError(f'Missing parameter for product {key}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, lb=1, vtype=GRB.INTEGER, name='')
    model.setObjective(gurobipy.quicksum((r_i[i] * x[i] for i in I)), GRB.MAXIMIZE)
    model.addConstrs((x[i] <= d_i[i] for i in I), name='')
    model.addConstrs((x[i] <= I_i[i] for i in I), name='')
    model.optimize()
    return model
m = solve_problem()