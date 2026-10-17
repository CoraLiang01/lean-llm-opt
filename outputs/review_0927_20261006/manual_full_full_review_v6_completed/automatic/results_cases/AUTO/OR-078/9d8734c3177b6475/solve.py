import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()

def to_float_series(df, col, idx):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
    try:
        return pd.Series(df[col].astype(float).values, index=idx)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to float: {e}")

def to_int_series(df, col, idx):
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
    try:
        return pd.Series(df[col].astype(int).values, index=idx)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to int: {e}")
cost_per_lot = to_float_series(energy_df, 'cost_per_lot', option_ids)
gen_per_lot = to_int_series(energy_df, 'gen_per_lot', option_ids)
m = gp.Model('Electricity_Procurement_Lot_MIP')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots per contract option:')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 0.5:
            tech = energy_df.loc[energy_df['option'] == oid, 'tech'].values[0]
            print(f'  {oid} ({tech}): {int(round(val))} lots')
    total_gen = sum((gen_per_lot[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')