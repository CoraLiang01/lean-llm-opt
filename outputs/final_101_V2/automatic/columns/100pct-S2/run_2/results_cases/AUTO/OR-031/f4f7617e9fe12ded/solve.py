import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
eligible_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].astype(str).str.casefold().str.strip()
eligible_df = energy_df[energy_df['tech_norm'].isin(eligible_techs)].copy()
option_ids = eligible_df['option'].astype(str).tolist()
if eligible_df['gen_per_lot'].isnull().any() or eligible_df['cost_per_lot'].isnull().any():
    raise ValueError("Missing values in 'gen_per_lot' or 'cost_per_lot' for eligible options.")
gen_per_lot = eligible_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = eligible_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
for oid in option_ids:
    if oid not in gen_per_lot or oid not in cost_per_lot:
        raise KeyError(f"Missing parameter for option '{oid}'.")
m = gp.Model('ElectricityProcurement')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x[oid] for oid in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Purchase plan (number of lots per option):')
    for oid in option_ids:
        val = x[oid].X
        if val > 1e-06:
            print(f"  {oid}: {int(round(val))} lots (tech: {eligible_df.loc[eligible_df['option'] == oid, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[oid]}, cost_per_lot: {cost_per_lot[oid]:.2f})")
    total_gen = sum((gen_per_lot[oid] * x[oid].X for oid in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')