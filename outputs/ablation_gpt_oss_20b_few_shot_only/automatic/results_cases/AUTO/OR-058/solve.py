import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    demand_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/demand.csv'
    fixed_cost_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/fixed_cost.csv'
    transport_path = 'Test_Dataset/Large-scale-or/UFLP_testing/UFLP1/transportation_costs.csv'
    demand_df = pd.read_csv(demand_path, dtype=str, keep_default_na=False, index_col=0)
    fixed_cost_df = pd.read_csv(fixed_cost_path, dtype=str, keep_default_na=False, index_col=0)
    transport_df = pd.read_csv(transport_path, dtype=str, keep_default_na=False, index_col=0)
    try:
        demand_series = demand_df['customer demand'].astype(float)
    except KeyError:
        raise KeyError("Column 'customer demand' not found in demand.csv")
    try:
        fixed_cost_series = fixed_cost_df['fixed_costs'].astype(float)
    except KeyError:
        raise KeyError("Column 'fixed_costs' not found in fixed_cost.csv")
    try:
        transport_df = transport_df.astype(float)
    except Exception as e:
        raise ValueError(f'Error converting transportation costs to float: {e}')
    suppliers = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
    stores = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    for s in suppliers:
        if s not in fixed_cost_series.index:
            raise ValueError(f'Missing fixed cost for supplier {s}')
    for t in stores:
        if t not in demand_series.index:
            raise ValueError(f'Missing demand for store {t}')
    for s in suppliers:
        if s not in transport_df.index:
            raise ValueError(f'Missing transportation data for supplier {s}')
        for t in stores:
            if t not in transport_df.columns:
                raise ValueError(f'Missing transportation cost for supplier {s} to store {t}')
    M = demand_series.sum()
    m = gurobipy.Model('UFLP')
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(suppliers, vtype=GRB.BINARY, name='')
    pairs = [(s, t) for s in suppliers for t in stores]
    x = m.addVars(pairs, lb=0, vtype=GRB.CONTINUOUS, name='')
    obj = gurobipy.quicksum((fixed_cost_series[s] * y[s] for s in suppliers)) + gurobipy.quicksum((transport_df.loc[s, t] * x[s, t] for s in suppliers for t in stores))
    m.setObjective(obj, GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[s, t] for s in suppliers)) == demand_series[t] for t in stores), name='')
    m.addConstrs((gurobipy.quicksum((x[s, t] for t in stores)) <= M * y[s] for s in suppliers), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()