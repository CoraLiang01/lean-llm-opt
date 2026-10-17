import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if not {'gen_per_lot', 'cost_per_lot', 'tech'}.issubset(energy_df.columns):
    missing = {'gen_per_lot', 'cost_per_lot', 'tech'} - set(energy_df.columns)
    raise KeyError(f'Missing required columns in energy.csv: {missing}')
option_keys = list(energy_df.index)
try:
    gen_per_lot = {}
    cost_per_lot = {}
    tech = {}
    for idx, row in energy_df.iterrows():
        option_key = idx
        gen_val = float(row['gen_per_lot'])
        cost_val = float(row['cost_per_lot'])
        tech_val = row['tech']
        gen_per_lot[option_key] = gen_val
        cost_per_lot[option_key] = cost_val
        tech[option_key] = tech_val
except Exception as e:
    raise ValueError(f'Error processing energy.csv: {e}')
m = gp.Model('Electricity_Lot_Sizing')
x_vars = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[k] * x_vars[k] for k in option_keys)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[k] * x_vars[k] for k in option_keys)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for k in option_keys:
        val = x_vars[k].X
        if val > 1e-06:
            print(f'  Option {k}: {int(round(val))} lots (tech: {tech[k]}, gen/lot: {gen_per_lot[k]}, cost/lot: {cost_per_lot[k]})')
    total_gen = sum((gen_per_lot[k] * x_vars[k].X for k in option_keys))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')