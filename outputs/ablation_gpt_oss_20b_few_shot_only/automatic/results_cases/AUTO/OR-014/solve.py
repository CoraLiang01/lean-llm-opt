import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in CSV")
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
    df['Demand'] = pd.to_numeric(df['Demand'], errors='raise', downcast='integer')
    df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise', downcast='integer')
    product_names = df['Product Name'].tolist()
    revenue = dict(zip(df['Product Name'], df['Revenue']))
    demand = dict(zip(df['Product Name'], df['Demand']))
    inventory = dict(zip(df['Product Name'], df['Initial Inventory']))
    for pname in product_names:
        if pname not in revenue or pname not in demand or pname not in inventory:
            raise ValueError(f"Missing parameter for product '{pname}'")
    m = gurobipy.Model('PizzaSales')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[i] * x[i] for i in product_names)), GRB.MAXIMIZE)
    for i in product_names:
        m.addConstr(x[i] <= demand[i], name=f'demand_{i}')
        m.addConstr(x[i] <= inventory[i], name=f'inventory_{i}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()