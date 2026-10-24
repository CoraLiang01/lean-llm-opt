import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ['Revenue', 'Demand', 'Initial Inventory']:
        df[col] = pd.to_numeric(df[col], errors='raise')
    product_names = df['Product Name'].tolist()
    if len(product_names) != len(set(product_names)):
        raise ValueError('Duplicate product names found')
    revenue_dict = dict(zip(product_names, df['Revenue']))
    demand_dict = dict(zip(product_names, df['Demand']))
    inventory_dict = dict(zip(product_names, df['Initial Inventory']))
    for i in product_names:
        if i not in revenue_dict or i not in demand_dict or i not in inventory_dict:
            raise ValueError(f'Missing parameter for {i}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.addConstrs((x[i] <= demand_dict[i] for i in product_names), name='')
    m.addConstrs((x[i] <= inventory_dict[i] for i in product_names), name='')
    m.setObjective(gurobipy.quicksum((revenue_dict[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()