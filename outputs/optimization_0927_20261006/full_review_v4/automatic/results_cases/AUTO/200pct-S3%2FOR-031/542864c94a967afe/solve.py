import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
energy_df = energy_df.set_index('option', drop=False)

def safe_int(series, col):
    try:
        return series[col].astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to int: {e}")

def safe_float(series, col):
    try:
        return series[col].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")
gen_per_lot_dict = safe_int(energy_df, 'gen_per_lot').to_dict()
cost_per_lot_dict = safe_float(energy_df, 'cost_per_lot').to_dict()
tech_dict = energy_df['tech'].to_dict()
for opt in option_ids:
    if opt not in gen_per_lot_dict or opt not in cost_per_lot_dict or opt not in tech_dict:
        raise KeyError(f"Missing required parameter for option '{opt}'.")
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots by contract option:')
    for opt in option_ids:
        val = x_vars[opt].X
        if val >= 1e-06:
            print(f'  Option: {opt:15s} | Tech: {tech_dict[opt]:12s} | Lots: {int(round(val))} | Gen/lot: {gen_per_lot_dict[opt]} | Cost/lot: {cost_per_lot_dict[opt]:.2f}')
    total_gen = sum((gen_per_lot_dict[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (Demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')