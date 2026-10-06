import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']

    def read_csv_with_encodings(path):
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/customer_demand.csv'
    supply_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/supply_capacity.csv'
    cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/TP_testing/TP7/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    supply_df = read_csv_with_encodings(supply_path)
    cost_df = read_csv_with_encodings(cost_path)
    demand_df.columns = demand_df.columns.str.strip()
    supply_df.columns = supply_df.columns.str.strip()
    cost_df.columns = cost_df.columns.str.strip()
    J = [str(j) for j in demand_df['customer']]
    I = [str(i) for i in supply_df['region']]
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = str(row['customer'])
        demand[key] = float(row['demand'])
    supply_capacity = {}
    for (_, row) in supply_df.iterrows():
        key = str(row['region'])
        supply_capacity[key] = float(row['supply_capacity'])
    cost = {}
    for (idx, row) in cost_df.iterrows():
        i = str(row.iloc[0])
        cost[i] = {}
        for j in J:
            if j not in cost_df.columns:
                raise ValueError(f'Store {j} not found in transportation_costs.csv columns.')
            val = row[j]
            if pd.isnull(val):
                raise ValueError(f'Missing cost for ({i},{j}) in transportation_costs.csv.')
            cost[i][j] = float(val)
    if set(I) != set(cost.keys()):
        raise ValueError('Mismatch between warehouses in supply_capacity.csv and transportation_costs.csv.')
    for i in I:
        if set(J) != set(cost[i].keys()):
            raise ValueError(f'Mismatch between stores in demand and cost for warehouse {i}.')
    m = gp.Model('GreenMart_TP')
    m.Params.MIPGap = 0.0001
    keys = [(i, j) for i in I for j in J]
    x = m.addVars(keys, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in I for j in J)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in I)) >= demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in J)) <= supply_capacity[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()