import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'].astype(str))
gen_per_lot = {}
cost_per_lot = {}
for (idx, row) in energy_df.iterrows():
    option = str(row['option'])
    if pd.isnull(row['gen_per_lot']) or pd.isnull(row['cost_per_lot']):
        raise ValueError(f'Missing gen_per_lot or cost_per_lot for option {option}')
    gen_per_lot[option] = int(row['gen_per_lot'])
    cost_per_lot[option] = float(row['cost_per_lot'])
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in options and parameter keys for gen_per_lot or cost_per_lot.')
demand = 200

def solve_problem():
    m = gp.Model('Electricity_Lot_Purchasing')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')