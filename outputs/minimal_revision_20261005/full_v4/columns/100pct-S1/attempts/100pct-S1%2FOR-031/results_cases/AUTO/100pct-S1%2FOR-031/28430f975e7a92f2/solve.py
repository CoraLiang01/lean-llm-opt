import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = list(energy_df['option'])
for col in ['gen_per_lot', 'cost_per_lot']:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
gen_per_lot = {}
cost_per_lot = {}
for (idx, row) in energy_df.iterrows():
    opt = str(row['option'])
    try:
        gen = int(row['gen_per_lot'])
        cost = float(row['cost_per_lot'])
    except Exception as e:
        raise ValueError(f'Invalid data in row {idx} for option {opt}: {e}')
    if opt in gen_per_lot or opt in cost_per_lot:
        raise ValueError(f'Duplicate option identifier found: {opt}')
    gen_per_lot[opt] = gen
    cost_per_lot[opt] = cost
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option keys between index set and parameter dictionaries.')
total_demand = 200

def solve_problem():
    m = gp.Model('Electricity_Procurement_MIP')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[opt] * x[opt] for opt in options)) >= total_demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')