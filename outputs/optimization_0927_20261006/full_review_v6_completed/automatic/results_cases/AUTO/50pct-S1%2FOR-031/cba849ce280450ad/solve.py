import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = energy_df['option'].tolist()
required_columns = ['gen_per_lot', 'cost_per_lot', 'tech']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")

def parse_int_series(series, colname):
    try:
        return series.astype(str).str.strip().astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")

def parse_float_series(series, colname):
    try:
        return series.astype(str).str.strip().astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
gen_per_lot = dict(zip(energy_df['option'], parse_int_series(energy_df['gen_per_lot'], 'gen_per_lot')))
cost_per_lot = dict(zip(energy_df['option'], parse_float_series(energy_df['cost_per_lot'], 'cost_per_lot')))
tech = dict(zip(energy_df['option'], energy_df['tech'].astype(str).str.strip()))
for i in options:
    if i not in gen_per_lot or i not in cost_per_lot or i not in tech:
        raise ValueError(f"Missing parameter for option '{i}'")
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in options)) >= 200, name='demand')
m.optimize()