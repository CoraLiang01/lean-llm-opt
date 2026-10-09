import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
valid_techs = {'coal', 'gas', 'renewables'}
energy_df = energy_df[energy_df['tech'].apply(lambda x: str(x).casefold().strip() in valid_techs)].copy()
option_ids = energy_df['option'].astype(str).tolist()
if not set(['option', 'gen_per_lot', 'cost_per_lot', 'tech']).issubset(energy_df.columns):
    raise KeyError('Missing required columns in energy.csv')
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
tech_of_option = energy_df.set_index('option')['tech'].astype(str).to_dict()
for oid in option_ids:
    if oid not in gen_per_lot or oid not in cost_per_lot or oid not in tech_of_option:
        raise ValueError(f'Missing parameter data for option {oid}')
m = gp.Model('Electricity_Lot_Purchasing')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x[oid] for oid in option_ids)) == 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Lot purchase plan (option_id, tech, lots, gen_per_lot, cost_per_lot):')
    for oid in option_ids:
        lots = x[oid].X
        if lots >= 1e-06:
            print(f'  {oid:15s}  {tech_of_option[oid]:12s}  {int(round(lots)):3d}  {gen_per_lot[oid]:6.1f}  {cost_per_lot[oid]:7.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')