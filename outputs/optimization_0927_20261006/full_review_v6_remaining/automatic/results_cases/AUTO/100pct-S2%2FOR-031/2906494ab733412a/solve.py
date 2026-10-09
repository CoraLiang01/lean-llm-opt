import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
energy_df = energy_df.set_index('option', drop=False)
for col in ['gen_per_lot', 'cost_per_lot']:
    try:
        energy_df[col] = pd.to_numeric(energy_df[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' in energy.csv contains non-numeric values: {e}")
option_ids = list(energy_df.index)
gen_per_lot = energy_df['gen_per_lot'].to_dict()
cost_per_lot = energy_df['cost_per_lot'].to_dict()
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for i in option_ids:
        xi = x_vars[i].X
        if xi > 1e-06:
            print(f"  Option {i}: {int(round(xi))} lots (tech: {energy_df.at[i, 'tech']}, gen_per_lot: {gen_per_lot[i]}, cost_per_lot: {cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * x_vars[i].X for i in option_ids))
    print(f'Total generation: {total_gen:.2f} (required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')