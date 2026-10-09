import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
eligible_techs = {'coal', 'gas', 'renewables'}
eligible_rows = energy_df['tech_norm'].isin({t.casefold() for t in eligible_techs})
options_df = energy_df.loc[eligible_rows].copy()
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in options_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv.")
options_df['option_id'] = options_df['option']
options_df = options_df.set_index('option_id', drop=False)
try:
    options_df['gen_per_lot'] = options_df['gen_per_lot'].astype(float)
    options_df['cost_per_lot'] = options_df['cost_per_lot'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting gen_per_lot or cost_per_lot to float: {e}')
option_ids = list(options_df.index)
gen_per_lot = options_df['gen_per_lot'].to_dict()
cost_per_lot = options_df['cost_per_lot'].to_dict()
m = gp.Model('ElectricityLotProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200.0
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for oid in option_ids:
        val = x_vars[oid].X
        if val > 1e-06:
            print(f"  Option {oid}: {int(round(val))} lots (tech: {options_df.at[oid, 'tech']}, gen_per_lot: {gen_per_lot[oid]}, cost_per_lot: {cost_per_lot[oid]:.2f})")
    total_gen = sum((gen_per_lot[oid] * x_vars[oid].X for oid in option_ids))
    print(f'Total generation scheduled: {total_gen:.2f} (demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')