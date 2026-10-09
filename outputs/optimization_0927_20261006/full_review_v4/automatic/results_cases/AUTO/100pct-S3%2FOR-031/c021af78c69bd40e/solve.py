import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
valid_techs = {'coal', 'gas', 'renewables'}
energy_df = energy_df[energy_df['tech_norm'].isin(valid_techs)].copy()
option_ids = energy_df['option'].tolist()
for col in ['gen_per_lot', 'cost_per_lot']:
    if energy_df[col].isnull().any() or (energy_df[col].str.strip() == '').any():
        raise ValueError(f"Missing or blank values in required column '{col}' for selected options.")
energy_df['gen_per_lot'] = energy_df['gen_per_lot'].astype(float)
energy_df['cost_per_lot'] = energy_df['cost_per_lot'].astype(float)
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
m = gp.Model('Electricity_Procurement_Lot_Selection')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='TotalDemand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots per contract option:')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen/lot: {gen_per_lot[opt]}, cost/lot: {cost_per_lot[opt]})")
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')