import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()

def to_int_series(series):
    return series.astype(str).str.strip().replace('', np.nan).astype(float).astype(int)

def to_float_series(series):
    return series.astype(str).str.strip().replace('', np.nan).astype(float)
gen_per_lot_dict = dict(zip(energy_df['option'], to_int_series(energy_df['gen_per_lot'])))
cost_per_lot_dict = dict(zip(energy_df['option'], to_float_series(energy_df['cost_per_lot'])))
if set(gen_per_lot_dict.keys()) != set(option_ids):
    raise ValueError('Mismatch in gen_per_lot keys and option IDs')
if set(cost_per_lot_dict.keys()) != set(option_ids):
    raise ValueError('Mismatch in cost_per_lot keys and option IDs')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot_dict[i] * x_vars[i] for i in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in option_ids:
        val = x_vars[i].X
        if val > 1e-06:
            print(f"  Option {i}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen_per_lot: {gen_per_lot_dict[i]}, cost_per_lot: {cost_per_lot_dict[i]:.2f})")
    total_gen = sum((gen_per_lot_dict[i] * x_vars[i].X for i in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')