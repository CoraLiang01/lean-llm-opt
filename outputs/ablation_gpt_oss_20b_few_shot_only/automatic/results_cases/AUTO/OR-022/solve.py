import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    csv_path = 'Test_Dataset/Large-scale-or/NRM_testing/NRM13/Salesorders.csv'
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    mask = df['Product Name'].str.contains('27in', na=False)
    df27 = df[mask].copy()
    try:
        revenue_series = pd.to_numeric(df27['Revenue'], errors='raise')
        demand_series = pd.to_numeric(df27['Demand'], errors='raise')
        inventory_series = pd.to_numeric(df27['Initial Inventory'], errors='raise')
    except Exception as e:
        raise ValueError(f'Numeric conversion error: {e}')
    product_names = df27['Product Name'].tolist()
    revenue_dict = dict(zip(product_names, revenue_series))
    demand_dict = dict(zip(product_names, demand_series))
    inventory_dict = dict(zip(product_names, inventory_series))
    for pname in product_names:
        if pname not in revenue_dict or pname not in demand_dict or pname not in inventory_dict:
            raise ValueError(f'Missing data for product: {pname}')
    m = gurobipy.Model('NRM13_27in')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(product_names, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='')
    for pname in product_names:
        m.addConstr(x[pname] <= demand_dict[pname], name=f'demand_{pname}')
    for pname in product_names:
        m.addConstr(x[pname] <= inventory_dict[pname], name=f'inventory_{pname}')
    m.setObjective(gurobipy.quicksum((revenue_dict[pname] * x[pname] for pname in product_names)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()