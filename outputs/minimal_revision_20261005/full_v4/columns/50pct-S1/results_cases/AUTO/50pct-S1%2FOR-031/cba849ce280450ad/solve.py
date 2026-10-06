import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'])
if energy_df['option'].isnull().any():
    raise ValueError('Missing option identifier(s) in energy.csv')
if energy_df['cost_per_lot'].isnull().any():
    raise ValueError('Missing cost_per_lot value(s) in energy.csv')
if energy_df['gen_per_lot'].isnull().any():
    raise ValueError('Missing gen_per_lot value(s) in energy.csv')
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
if set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in cost_per_lot keys and options')
if set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in gen_per_lot keys and options')
demand = 200
m = gp.Model('generation_lot_selection')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= demand, name='demand')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')