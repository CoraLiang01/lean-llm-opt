import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'])
if len(options) != len(energy_df):
    raise ValueError('Duplicate or missing option identifiers in energy.csv.')
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
tech = energy_df.set_index('option')['tech'].astype(str).to_dict()
for opt in options:
    if opt not in gen_per_lot or opt not in cost_per_lot or opt not in tech:
        raise KeyError(f'Missing parameter for option {opt} in energy.csv.')
m = gp.Model('Electricity_Procurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x[opt] for opt in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected contract lots (option, tech, lots, total_gen, total_cost):')
    for opt in options:
        lots = x[opt].X
        if lots >= 1e-06:
            total_gen = gen_per_lot[opt] * lots
            total_cost = cost_per_lot[opt] * lots
            print(f'  {opt:15s} {tech[opt]:12s} {int(round(lots)):4d} {total_gen:7.2f} {total_cost:9.2f}')
    total_generation = sum((gen_per_lot[opt] * x[opt].X for opt in options))
    print(f'Total generation provided: {total_generation:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')