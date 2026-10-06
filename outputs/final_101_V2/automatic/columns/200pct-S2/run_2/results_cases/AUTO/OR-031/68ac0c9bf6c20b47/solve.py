import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
if not set(['option', 'gen_per_lot', 'cost_per_lot', 'tech']).issubset(energy_df.columns):
    raise KeyError('Missing required columns in energy.csv')
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
tech = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
m = gp.Model('ElectricityLotProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan (option, tech, lots purchased, generation, cost):')
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f'  {i:15s} | {tech[i]:11s} | {int(round(xi)):3d} lots | {gen_per_lot[i] * int(round(xi)):6.1f} gen | {cost_per_lot[i] * int(round(xi)):8.2f} cost')
    total_gen = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation: {total_gen:.2f} (demand required: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')