import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()

def to_int_series(df, col):
    return df[col].astype(int).to_dict()

def to_float_series(df, col):
    return df[col].astype(float).to_dict()
gen_per_lot = dict(zip(option_ids, energy_df['gen_per_lot'].astype(int)))
cost_per_lot = dict(zip(option_ids, energy_df['cost_per_lot'].astype(float)))
tech = dict(zip(option_ids, energy_df['tech']))
if set(gen_per_lot.keys()) != set(option_ids) or set(cost_per_lot.keys()) != set(option_ids):
    raise ValueError('Mismatch in parameter coverage for generation options.')
demand = 200
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 1e-06:
            print(f'  Option {oid} (tech={tech[oid]}): {int(round(val))} lots, gen_per_lot={gen_per_lot[oid]}, cost_per_lot={cost_per_lot[oid]:.2f}')
    total_gen = sum((gen_per_lot[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand={demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')