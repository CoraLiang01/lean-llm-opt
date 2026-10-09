import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = {row['option']: int(row['gen_per_lot']) for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to int: {e}")
try:
    cost_per_lot = {row['option']: float(row['cost_per_lot']) for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
tech_type = {row['option']: row['tech'] for (_, row) in energy_df.iterrows()}
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    tech_groups = {}
    for opt in option_ids:
        tech = tech_type[opt]
        tech_groups.setdefault(tech, []).append(opt)
    for tech in sorted(tech_groups):
        print(f'\n{tech.capitalize()} options:')
        for opt in tech_groups[tech]:
            val = x_vars[opt].X
            if val > 0.5:
                print(f'  {opt}: {int(round(val))} lots (gen_per_lot={gen_per_lot[opt]}, cost_per_lot={cost_per_lot[opt]:.2f})')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'\nTotal generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')