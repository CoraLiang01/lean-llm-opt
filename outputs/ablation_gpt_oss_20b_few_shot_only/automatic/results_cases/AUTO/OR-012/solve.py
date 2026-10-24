import pandas as pd
import gurobipy
csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df['Revenue'] = pd.to_numeric(df['Revenue'], errors='raise')
df['Demand'] = pd.to_numeric(df['Demand'], errors='raise')
df['Initial Inventory'] = pd.to_numeric(df['Initial Inventory'], errors='raise')
products = df['Product Name'].tolist()
revenue = dict(zip(products, df['Revenue']))
demand = dict(zip(products, df['Demand']))
init_inv = dict(zip(products, df['Initial Inventory']))
for p in products:
    if p not in revenue or p not in demand or p not in init_inv:
        raise ValueError(f'Missing parameter for product {p}')

def solve_problem():
    m = gurobipy.Model('MaxRevenue')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=gurobipy.GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in products)), gurobipy.GRB.MAXIMIZE)
    for p in products:
        m.addConstr(x[p] <= demand[p], name='')
        m.addConstr(x[p] <= init_inv[p], name='')
    m.optimize()
    return m
m = solve_problem()
if m.Status == gurobipy.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')