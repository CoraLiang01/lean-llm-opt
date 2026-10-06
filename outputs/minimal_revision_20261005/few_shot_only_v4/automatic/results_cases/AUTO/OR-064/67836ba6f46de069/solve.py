import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():

    def read_csv_robust(path, **kwargs):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']:
            try:
                return pd.read_csv(path, encoding=enc, **kwargs)
            except UnicodeDecodeError:
                continue
        raise RuntimeError(f'Could not decode {path} with tried encodings.')
    demand_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/demand.csv'
    fixed_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/fixed_cost.csv'
    trans_cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/UFLP_testing/UFLP8/transportation_costs.csv'
    demand_df = read_csv_robust(demand_path)
    if not {'customer', 'demand'}.issubset(demand_df.columns):
        raise ValueError("demand.csv must have columns 'customer' and 'demand'")
    demand_df['customer'] = demand_df['customer'].astype(str)
    customers = demand_df['customer'].tolist()
    demand = dict(zip(demand_df['customer'], demand_df['demand']))
    fixed_df = read_csv_robust(fixed_cost_path)
    if not {'Unnamed: 0', 'fixed_costs'}.issubset(fixed_df.columns):
        raise ValueError("fixed_cost.csv must have columns 'Unnamed: 0' and 'fixed_costs'")
    fixed_df['Unnamed: 0'] = fixed_df['Unnamed: 0'].astype(str)
    suppliers = fixed_df['Unnamed: 0'].tolist()
    fixed_cost = dict(zip(fixed_df['Unnamed: 0'], fixed_df['fixed_costs']))
    trans_df = read_csv_robust(trans_cost_path)
    if 'Unnamed: 0' not in trans_df.columns:
        raise ValueError("transportation_costs.csv must have column 'Unnamed: 0'")
    trans_df['Unnamed: 0'] = trans_df['Unnamed: 0'].astype(str)
    for c in customers:
        if c not in trans_df.columns:
            raise ValueError(f'transportation_costs.csv missing column {c}')
    cost = {}
    for (_, row) in trans_df.iterrows():
        i = row['Unnamed: 0']
        if i not in suppliers:
            continue
        cost[i] = {}
        for j in customers:
            cost[i][j] = row[j]
    for i in suppliers:
        if i not in cost:
            raise ValueError(f'Supplier {i} missing in transportation_costs.csv')
        for j in customers:
            if j not in cost[i]:
                raise ValueError(f'Cost for supplier {i}, customer {j} missing in transportation_costs.csv')
    for j in customers:
        if j not in demand:
            raise ValueError(f'Customer {j} missing in demand.csv')
    for i in suppliers:
        if i not in fixed_cost:
            raise ValueError(f'Supplier {i} missing in fixed_cost.csv')
    M = sum((demand[j] for j in customers))
    m = gp.Model('UFLP8')
    m.Params.MIPGap = 0.0001
    x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)) + gp.quicksum((fixed_cost[i] * y[i] for i in suppliers)), GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= M * y[i] for i in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()