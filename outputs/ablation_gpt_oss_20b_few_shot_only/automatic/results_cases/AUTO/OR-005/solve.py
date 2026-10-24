import pandas as pd
import gurobipy as gp
from gurobipy import GRB
customer_demand_path = 'Test_Dataset/Large-scale-or/TP_testing/TP5/customer_demand.csv'
supply_capacity_path = 'Test_Dataset/Large-scale-or/TP_testing/TP5/supply_capacity.csv'
transportation_costs_path = 'Test_Dataset/Large-scale-or/TP_testing/TP5/transportation_costs.csv'

def read_customer_demand(path):
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError('customer_demand.csv format unexpected')
    demand = {}
    for (_, row) in df.iterrows():
        demand_id = row[0]
        val = float(row[1])
        demand[demand_id] = val
    return demand

def read_supply_capacity(path):
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if df.shape[1] < 2:
        raise ValueError('supply_capacity.csv format unexpected')
    supply = {}
    for (_, row) in df.iterrows():
        supplier_id = row[0]
        val = float(row[1])
        supply[supplier_id] = val
    return supply

def read_transportation_costs(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    demand_cols = df.columns[1:]
    cost = {}
    for (_, row) in df.iterrows():
        supplier_id = row[0]
        cost[supplier_id] = {}
        for col in demand_cols:
            val = float(row[col])
            cost[supplier_id][col] = val
    return cost

def solve_problem():
    demand = read_customer_demand(customer_demand_path)
    supply = read_supply_capacity(supply_capacity_path)
    cost = read_transportation_costs(transportation_costs_path)
    suppliers = ['supplier1', 'supplier2', 'supplier3', 'supplier4', 'supplier5', 'supplier6', 'supplier7', 'supplier8']
    customers = ['demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8']
    for s in suppliers:
        if s not in supply:
            raise ValueError(f'Missing supply capacity for {s}')
    for c in customers:
        if c not in demand:
            raise ValueError(f'Missing demand for {c}')
    for s in suppliers:
        if s not in cost:
            raise ValueError(f'Missing cost data for supplier {s}')
        for c in customers:
            if c not in cost[s]:
                raise ValueError(f'Missing cost for supplier {s} to customer {c}')
    m = gp.Model('TransportationProblem')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in suppliers for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) >= demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()