import gurobipy as gp
import pandas as pd
import numpy as np
import math
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
option_ids = energy_df['option'].astype(str).tolist()
required_cols = ['option', 'cost_per_lot', 'gen_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
    if energy_df[col].isnull().any():
        raise ValueError(f"Column '{col}' contains missing values in energy.csv")
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
m = gp.Model('Electricity_Procurement')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in option_ids)) >= 200, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Contract option purchase plan (number of lots):')
    for i in option_ids:
        xi = x[i].X
        if xi >= 1e-06:
            print(f"  {i}: {int(round(xi))} lot(s) (tech: {energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[i]}, cost_per_lot: {cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * x[i].X for i in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')