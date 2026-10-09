import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Required column 'option' not found in energy.csv")
option_ids = energy_df['option'].tolist()
energy_df = energy_df.set_index('option', drop=False)

def safe_int(series, col):
    try:
        return series[col].astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to int: {e}")

def safe_float(series, col):
    try:
        return series[col].astype(float)
    except Exception as e:
        raise ValueError(f"Column '{col}' could not be converted to float: {e}")
if 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Required column 'gen_per_lot' not found in energy.csv")
gen_per_lot_dict = safe_int(energy_df, 'gen_per_lot').to_dict()
if 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Required column 'cost_per_lot' not found in energy.csv")
cost_per_lot_dict = safe_float(energy_df, 'cost_per_lot').to_dict()
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='Demand')
m.optimize()