import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()

def to_float_series(s):
    return pd.to_numeric(s, errors='raise')

def to_int_series(s):
    return pd.to_numeric(s, errors='raise', downcast='integer')
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv")
energy_df = energy_df.set_index('option', drop=False)
gen_per_lot = to_int_series(energy_df['gen_per_lot']).to_dict()
cost_per_lot = to_float_series(energy_df['cost_per_lot']).to_dict()
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  {opt}: {int(round(val))} lots (tech: {energy_df.at[opt, 'tech']}, gen_per_lot: {gen_per_lot[opt]}, cost_per_lot: {cost_per_lot[opt]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')