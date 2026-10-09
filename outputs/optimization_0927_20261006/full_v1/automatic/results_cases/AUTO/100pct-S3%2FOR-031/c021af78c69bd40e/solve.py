import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
valid_techs = {'coal', 'gas', 'renewables'}
energy_df = energy_df[energy_df['tech'].str.casefold().isin({t.casefold() for t in valid_techs})].copy()
option_keys = energy_df['option'].tolist()
if len(option_keys) != len(set(option_keys)):
    raise ValueError("Duplicate option identifiers found in 'option' column.")

def to_float_col(df, col):
    try:
        return df[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")

def to_int_col(df, col):
    try:
        return df[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")
cost_per_lot = dict(zip(option_keys, to_float_col(energy_df, 'cost_per_lot')))
gen_per_lot = dict(zip(option_keys, to_int_col(energy_df, 'gen_per_lot')))
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_keys)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_keys)) >= total_demand, name='demand')
m.optimize()