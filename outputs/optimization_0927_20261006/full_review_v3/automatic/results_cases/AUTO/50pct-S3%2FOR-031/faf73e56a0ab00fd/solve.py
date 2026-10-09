import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'cost_per_lot' to float for all options: {e}")
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'gen_per_lot' to float for all options: {e}")
tech = energy_df.set_index('option')['tech'].to_dict()
m = gp.Model('Electricity_Lot_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots by technology:')
    tech_groups = {}
    for opt in option_ids:
        t = tech[opt]
        tech_groups.setdefault(t, []).append(opt)
    for t in sorted(tech_groups):
        print(f'  {t.capitalize()}:')
        for opt in tech_groups[t]:
            val = x_vars[opt].X
            if val >= 1e-06:
                print(f'    {opt}: {int(round(val))} lot(s), gen_per_lot={gen_per_lot[opt]}, cost_per_lot={cost_per_lot[opt]}')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')