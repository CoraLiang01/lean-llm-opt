import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
if energy_df['gen_per_lot'].isnull().any():
    raise ValueError("Missing values in 'gen_per_lot' column.")
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
if energy_df['cost_per_lot'].isnull().any():
    raise ValueError("Missing values in 'cost_per_lot' column.")
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
tech = energy_df.set_index('option')['tech'].astype(str).to_dict()
if set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in gen_per_lot keys and options.')
if set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in cost_per_lot keys and options.')
m = gp.Model('Electricity_Lot_Purchasing')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan (option, tech, lots, total_gen, total_cost):')
    for i in options:
        lots = x[i].X
        if lots >= 1e-06:
            total_gen = gen_per_lot[i] * lots
            total_cost = cost_per_lot[i] * lots
            print(f'  {i:15s} | {tech[i]:11s} | {int(round(lots)):4d} lots | {total_gen:7.2f} gen | {total_cost:9.2f} cost')
    total_generation = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation purchased: {total_generation:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')