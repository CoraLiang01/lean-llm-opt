import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
if energy_df['option'].duplicated().any():
    raise ValueError('Duplicate option identifiers found in energy.csv.')
options = list(energy_df['option'])
cost_per_lot = {}
gen_per_lot = {}
tech = {}
for (idx, row) in energy_df.iterrows():
    opt = str(row['option'])
    cost = float(row['cost_per_lot'])
    gen = int(row['gen_per_lot'])
    t = str(row['tech'])
    cost_per_lot[opt] = cost
    gen_per_lot[opt] = gen
    tech[opt] = t
for opt in options:
    if opt not in cost_per_lot or opt not in gen_per_lot:
        raise ValueError(f'Missing cost or generation data for option {opt}.')
demand = 200

def solve_problem():
    m = gp.Model('Electricity_Lot_Procurement')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[opt] * x[opt] for opt in options)) >= demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')