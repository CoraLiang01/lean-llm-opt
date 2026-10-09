import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
if len(option_ids) != len(set(option_ids)):
    raise ValueError("Duplicate option identifiers found in 'option' column.")

def to_int_series(series):
    return series.str.strip().astype(int)

def to_float_series(series):
    return series.str.strip().astype(float)
gen_per_lot = dict(zip(option_ids, to_int_series(energy_df['gen_per_lot'])))
cost_per_lot = dict(zip(option_ids, to_float_series(energy_df['cost_per_lot'])))
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()