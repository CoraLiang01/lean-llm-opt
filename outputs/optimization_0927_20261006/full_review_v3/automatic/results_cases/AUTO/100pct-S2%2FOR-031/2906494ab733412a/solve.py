import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
energy_df = energy_df.set_index('option', drop=False)
required_cols = ['gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
for col in required_cols:
    try:
        energy_df[col] = pd.to_numeric(energy_df[col], errors='raise')
    except Exception as e:
        raise ValueError(f"Column '{col}' in energy.csv could not be converted to numeric: {e}")
options = list(energy_df.index)
gen_per_lot = energy_df['gen_per_lot'].to_dict()
cost_per_lot = energy_df['cost_per_lot'].to_dict()
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in options:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {energy_df.at[opt, 'tech']}, gen_per_lot: {gen_per_lot[opt]}, cost_per_lot: {cost_per_lot[opt]:.2f})")
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in options))
    print(f'Total generation purchased: {total_gen:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')