import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = energy_df['option'].tolist()
required_columns = ['gen_per_lot', 'cost_per_lot']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")

def safe_int(series, colname):
    try:
        return series.astype(str).str.strip().astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")

def safe_float(series, colname):
    try:
        return series.astype(str).str.strip().astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")
gen_per_lot = dict(zip(energy_df['option'], safe_int(energy_df['gen_per_lot'], 'gen_per_lot')))
cost_per_lot = dict(zip(energy_df['option'], safe_float(energy_df['cost_per_lot'], 'cost_per_lot')))
m = gp.Model('Electricity_Procurement_Lot_Selection')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) >= total_demand, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots per option:')
    for opt in options:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[opt]}, cost_per_lot: {cost_per_lot[opt]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')