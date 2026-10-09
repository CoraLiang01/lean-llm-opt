import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()

def to_float_series(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")

def to_int_series(df, col):
    try:
        return df[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")
cost_per_lot_dict = dict(zip(option_ids, to_float_series(energy_df, 'cost_per_lot')))
gen_per_lot_dict = dict(zip(option_ids, to_int_series(energy_df, 'gen_per_lot')))
if set(cost_per_lot_dict.keys()) != set(option_ids) or set(gen_per_lot_dict.keys()) != set(option_ids):
    raise ValueError('Mismatch in parameter keys and option IDs.')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f'  Option {opt}: {int(round(val))} lots (gen_per_lot={gen_per_lot_dict[opt]}, cost_per_lot={cost_per_lot_dict[opt]:.2f})')
    total_gen = sum((gen_per_lot_dict[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')