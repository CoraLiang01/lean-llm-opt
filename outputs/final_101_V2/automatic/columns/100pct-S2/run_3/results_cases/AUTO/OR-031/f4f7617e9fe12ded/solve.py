import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
eligible_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].astype(str).str.strip().str.casefold()
eligible_techs_norm = set([t.casefold() for t in eligible_techs])
eligible_df = energy_df[energy_df['tech_norm'].isin(eligible_techs_norm)].copy()
option_ids = eligible_df['option'].astype(str).tolist()
if not set(['option', 'gen_per_lot', 'cost_per_lot']).issubset(eligible_df.columns):
    raise KeyError('Missing required columns in energy.csv.')
gen_per_lot = eligible_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = eligible_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option IDs and parameter keys.')
m = gp.Model('ElectricityProcurement')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan (option_id: lots purchased):')
    for i in option_ids:
        xi = int(round(x[i].X))
        if xi > 0:
            print(f"  {i}: {xi} lot(s) (tech={eligible_df.loc[eligible_df['option'] == i, 'tech'].values[0]}, gen_per_lot={gen_per_lot[i]}, cost_per_lot={cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * int(round(x[i].X)) for i in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')