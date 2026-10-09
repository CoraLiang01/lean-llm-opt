import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(set(option_ids)):
    raise ValueError("Duplicate option IDs found in 'option' column.")
try:
    gen_per_lot = pd.to_numeric(energy_df.set_index('option')['gen_per_lot']).to_dict()
    cost_per_lot = pd.to_numeric(energy_df.set_index('option')['cost_per_lot']).to_dict()
except Exception as e:
    raise ValueError(f'Error converting numeric fields: {e}')
for oid in option_ids:
    if oid not in gen_per_lot or oid not in cost_per_lot:
        raise KeyError(f"Missing parameter for option '{oid}'.")
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots per contract option:')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 0.5:
            print(f'  {oid}: {int(round(val))} lots')
    total_gen = sum((gen_per_lot[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')