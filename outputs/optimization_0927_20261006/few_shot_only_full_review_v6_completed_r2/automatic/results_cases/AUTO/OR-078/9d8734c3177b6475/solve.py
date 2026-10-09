import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = list(energy_df['option'])

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values or missing data.") from e
if 'gen_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'gen_per_lot' in energy.csv")
if 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required column 'cost_per_lot' in energy.csv")
gen_per_lot = dict(zip(energy_df['option'], safe_float(energy_df['gen_per_lot'], 'gen_per_lot')))
cost_per_lot = dict(zip(energy_df['option'], safe_float(energy_df['cost_per_lot'], 'cost_per_lot')))
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
demand = 200.0
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) == demand, name='Demand')
m.optimize()