import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for idx, row in energy_df.iterrows():
        option = row['option']
        try:
            gen_per_lot[option] = int(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Invalid gen_per_lot for option {option}: {row['gen_per_lot']}")
        try:
            cost_per_lot[option] = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Invalid cost_per_lot for option {option}: {row['cost_per_lot']}")
except KeyError as e:
    raise KeyError(f'Missing required column in CSV: {e}')
m = gp.Model('Electricity_Procurement_MIP')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Selected lots per option (nonzero only):')
    for i in option_ids:
        val = x_vars[i].X
        if val > 1e-06:
            print(f"  {i}: {int(round(val))} lot(s) (tech: {energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[i]}, cost_per_lot: {cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * x_vars[i].X for i in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')