import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].astype(str).tolist()

def to_float_series(df, col):
    return pd.to_numeric(df[col], errors='raise')
cost_per_lot = dict(zip(energy_df['option'].astype(str), to_float_series(energy_df, 'cost_per_lot')))
gen_per_lot = dict(zip(energy_df['option'].astype(str), to_float_series(energy_df, 'gen_per_lot')))
if set(option_ids) != set(cost_per_lot.keys()) or set(option_ids) != set(gen_per_lot.keys()):
    raise ValueError('Mismatch in option index set and parameter keys.')
m = gp.Model('ElectricityLotProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()