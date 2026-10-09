import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
eligible_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
eligible_techs_norm = {t.casefold() for t in eligible_techs}
eligible_mask = energy_df['tech_norm'].isin(eligible_techs_norm)
eligible_df = energy_df[eligible_mask].copy()
option_ids = eligible_df['option'].tolist()
for col in ['gen_per_lot', 'cost_per_lot']:
    if col not in eligible_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv")
    if not np.all(eligible_df[col].str.strip().replace('', np.nan).notnull()):
        raise ValueError(f"Missing or blank values found in column '{col}' for eligible options.")
gen_per_lot = eligible_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = eligible_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {eligible_df.loc[eligible_df['option'] == opt, 'tech'].iloc[0]}, gen/lot: {gen_per_lot[opt]}, cost/lot: {cost_per_lot[opt]})")
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')