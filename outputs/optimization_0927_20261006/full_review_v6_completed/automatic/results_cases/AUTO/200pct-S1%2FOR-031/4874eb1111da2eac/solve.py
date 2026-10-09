import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
if 'gen_per_lot' not in energy_df.columns or 'cost_per_lot' not in energy_df.columns:
    raise KeyError("Missing required columns 'gen_per_lot' or 'cost_per_lot' in energy.csv")
try:
    gen_per_lot_dict = dict(zip(option_ids, energy_df['gen_per_lot'].astype(int)))
    cost_per_lot_dict = dict(zip(option_ids, energy_df['cost_per_lot'].astype(float)))
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' or 'cost_per_lot' to numeric: {e}")
total_demand = 200
m = gp.Model('Electricity_Procurement_Lot_MIP')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[oid] * x_vars[oid] for oid in option_ids)) >= total_demand, name='TotalDemand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots to purchase:')
    for oid in option_ids:
        val = x_vars[oid].X
        if val >= 1e-06:
            print(f"  Option {oid}: {int(round(val))} lot(s), Tech: {energy_df.loc[energy_df['option'] == oid, 'tech'].values[0]}, Gen/lot: {gen_per_lot_dict[oid]}, Cost/lot: {cost_per_lot_dict[oid]:.2f}")
    total_gen = sum((gen_per_lot_dict[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')