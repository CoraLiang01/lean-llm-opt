import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_try_encodings(path):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP6/transportation_costs.csv'
    demand_df = read_csv_try_encodings(demand_path)
    fixed_cost_df = read_csv_try_encodings(fixed_cost_path)
    trans_cost_df = read_csv_try_encodings(trans_cost_path)
    suppliers = list(fixed_cost_df['Unnamed: 0'])
    stores = [col for col in trans_cost_df.columns if col != 'Unnamed: 0']
    if len(demand_df) != len(stores):
        raise ValueError('Mismatch between number of stores in demand.csv and transportation_costs.csv')
    demand = dict(zip(stores, demand_df['demand']))
    fixed_cost = dict(zip(fixed_cost_df['Unnamed: 0'], fixed_cost_df['fixed_costs']))
    cost = {}
    for (idx, row) in trans_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for store in stores:
            cost[supplier][store] = row[store]
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in stores:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, store {j}')
    for j in stores:
        if j not in demand:
            raise ValueError(f'Missing demand for store {j}')
    M = sum((demand[j] for j in stores))
    m = gp.Model('Iowa_Facility_Location')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(suppliers, stores, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in stores)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in stores), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in stores)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()