import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_with_encodings(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP5/transportation_costs.csv'
    demand_df = read_csv_with_encodings(demand_path)
    fixed_cost_df = read_csv_with_encodings(fixed_cost_path)
    trans_cost_df = read_csv_with_encodings(trans_cost_path)
    suppliers = list(fixed_cost_df['Unnamed: 0'])
    branches = list(demand_df['customer'])
    demand = {}
    for (_, row) in demand_df.iterrows():
        key = row['customer']
        if key in demand:
            demand[key] += row['demand']
        else:
            demand[key] = row['demand']
    fixed_cost = {}
    for (_, row) in fixed_cost_df.iterrows():
        key = row['Unnamed: 0']
        if key in fixed_cost:
            fixed_cost[key] += row['fixed_costs']
        else:
            fixed_cost[key] = row['fixed_costs']
    cost = {}
    for (_, row) in trans_cost_df.iterrows():
        supplier = row['Unnamed: 0']
        if supplier not in suppliers:
            continue
        cost[supplier] = {}
        for branch in branches:
            if branch not in trans_cost_df.columns:
                raise ValueError(f'Branch {branch} not found in transportation_costs.csv columns.')
            val = row[branch]
            cost[supplier][branch] = val
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Missing fixed cost for supplier {i}')
        if i not in cost:
            raise ValueError(f'Missing transportation cost row for supplier {i}')
        for j in branches:
            if j not in cost[i]:
                raise ValueError(f'Missing transportation cost for supplier {i}, branch {j}')
    for j in branches:
        if j not in demand:
            raise ValueError(f'Missing demand for branch {j}')
    M = sum((demand[j] for j in branches))
    m = gp.Model('UFLP5')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, branches, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in branches)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in branches), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in branches)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()