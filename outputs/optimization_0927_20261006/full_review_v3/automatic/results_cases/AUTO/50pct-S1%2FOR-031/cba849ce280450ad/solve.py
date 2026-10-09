import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(set(option_ids)):
    raise ValueError("Duplicate option IDs found in 'option' column.")
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric fields: {e}')
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option IDs and parameter keys.')
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            tech = energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]
            print(f'  Option: {opt} (Tech: {tech}), Lots: {int(round(val))}, Gen/lot: {gen_per_lot[opt]}, Cost/lot: {cost_per_lot[opt]:.2f}')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation: {total_gen:.2f} (Demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')