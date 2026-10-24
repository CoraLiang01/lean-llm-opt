import pandas as pd
import gurobipy
from gurobipy import GRB

def solve_problem():
    energy_path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
    df = pd.read_csv(energy_path, dtype=str, keep_default_na=False)
    try:
        df['gen_per_lot'] = df['gen_per_lot'].astype(float)
        df['cost_per_lot'] = df['cost_per_lot'].astype(float)
    except KeyError as e:
        raise KeyError(f'Missing required column in CSV: {e}')
    options = df['option'].tolist()
    gen = dict(zip(options, df['gen_per_lot']))
    cost = dict(zip(options, df['cost_per_lot']))
    for opt in options:
        if opt not in gen or opt not in cost:
            raise ValueError(f'Missing parameters for option {opt}')
    D = 200.0
    m = gurobipy.Model('EnergyLotSelection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(options, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gurobipy.quicksum((cost[i] * x[i] for i in options)), GRB.MINIMIZE)
    m.addConstr(gurobipy.quicksum((gen[i] * x[i] for i in options)) >= D, name='Demand')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective value: {m.ObjVal:.4f}')
        for var in m.getVars():
            print(f'{var.VarName} = {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()