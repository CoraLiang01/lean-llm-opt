import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
eligible_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
eligible_df = energy_df[energy_df['tech_norm'].isin({t.casefold() for t in eligible_techs})].copy()
required_columns = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_columns:
    if col not in eligible_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv.")
eligible_df['option'] = eligible_df['option'].astype(str)
if eligible_df['option'].duplicated().any():
    raise ValueError("Duplicate 'option' identifiers found in eligible options.")
try:
    eligible_df['gen_per_lot'] = eligible_df['gen_per_lot'].astype(float)
    eligible_df['cost_per_lot'] = eligible_df['cost_per_lot'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting numeric fields: {e}')
option_ids = eligible_df['option'].tolist()
gen_per_lot = dict(zip(option_ids, eligible_df['gen_per_lot']))
cost_per_lot = dict(zip(option_ids, eligible_df['cost_per_lot']))
m = gp.Model('Electricity_Procurement_Lot_Selection')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='TotalDemand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('--- Lot Purchase Plan ---')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {eligible_df.loc[eligible_df['option'] == opt, 'tech'].values[0]}, gen/lot: {gen_per_lot[opt]}, cost/lot: {cost_per_lot[opt]:.2f})")
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')