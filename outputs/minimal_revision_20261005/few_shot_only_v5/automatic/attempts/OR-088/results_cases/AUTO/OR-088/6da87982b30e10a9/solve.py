import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/cost.csv'
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP12/demand.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            cost_df = pd.read_csv(cost_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {cost_path} with tried encodings.')
    for enc in encodings:
        try:
            demand_df = pd.read_csv(demand_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError(f'Could not decode {demand_path} with tried encodings.')
    plant_col = 'plant'
    customer_cols = [col for col in cost_df.columns if col.startswith('C')]
    plants = cost_df[plant_col].astype(str).tolist()
    customers = [col for col in customer_cols]
    if not set(['fixed_cost', 'capacity']).issubset(cost_df.columns):
        raise ValueError("cost.csv must contain 'fixed_cost' and 'capacity' columns.")
    fixed_cost = dict(zip(cost_df[plant_col].astype(str), cost_df['fixed_cost']))
    capacity = dict(zip(cost_df[plant_col].astype(str), cost_df['capacity']))
    cost = {}
    for (_, row) in cost_df.iterrows():
        i = str(row[plant_col])
        cost[i] = {}
        for j in customers:
            cost[i][j] = row[j]
    if not set(['customer', 'demand']).issubset(demand_df.columns):
        raise ValueError("demand.csv must contain 'customer' and 'demand' columns.")
    demand = {}
    for (_, row) in demand_df.iterrows():
        j = str(row['customer'])
        demand[j] = row['demand']
    if set(customers) != set(demand.keys()):
        raise ValueError(f'Mismatch between customers in cost.csv ({customers}) and demand.csv ({list(demand.keys())})')
    if set(plants) != set(fixed_cost.keys()) or set(plants) != set(capacity.keys()) or set(plants) != set(cost.keys()):
        raise ValueError('Mismatch in plant identifiers between cost.csv columns.')
    for i in plants:
        if set(cost[i].keys()) != set(customers):
            raise ValueError(f'Plant {i} missing cost entries for some customers.')
    m = gp.Model('UFLP')
    m.Params.MIPGap = 0.0001
    x_keys = [(i, j) for i in plants for j in customers]
    x = m.addVars(x_keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(plants, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()