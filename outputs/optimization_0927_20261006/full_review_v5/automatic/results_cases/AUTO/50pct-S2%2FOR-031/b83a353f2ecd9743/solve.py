import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
required_columns = ['gen_per_lot', 'cost_per_lot']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")

def to_int_strict(x):
    try:
        return int(x)
    except Exception:
        raise ValueError(f'Invalid integer value: {x}')

def to_float_strict(x):
    try:
        return float(x)
    except Exception:
        raise ValueError(f'Invalid float value: {x}')
energy_df['gen_per_lot_num'] = energy_df['gen_per_lot'].apply(to_int_strict)
energy_df['cost_per_lot_num'] = energy_df['cost_per_lot'].apply(to_float_strict)
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot_num']))
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot_num']))
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()