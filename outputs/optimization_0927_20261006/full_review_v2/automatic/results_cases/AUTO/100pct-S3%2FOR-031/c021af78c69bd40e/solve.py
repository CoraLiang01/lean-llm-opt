import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)

def normalize_tech(val):
    return val.strip().casefold()
valid_techs = {'coal', 'gas', 'renewables'}
energy_df = energy_df[energy_df['tech'].apply(lambda x: normalize_tech(x) in valid_techs)].copy()
option_ids = energy_df['option'].tolist()
for col in ['gen_per_lot', 'cost_per_lot']:
    if not all(energy_df[col].apply(lambda x: re.fullmatch('^\\s*-?\\d+(\\.\\d+)?\\s*$', x))):
        raise ValueError(f"Non-numeric or missing values found in column '{col}' for selected options.")
gen_per_lot = {row['option']: int(float(row['gen_per_lot'])) for (_, row) in energy_df.iterrows()}
cost_per_lot = {row['option']: float(row['cost_per_lot']) for (_, row) in energy_df.iterrows()}
m = gp.Model('Electricity_Lot_Procurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for opt in option_ids:
        val = x_vars[opt].X
        if val >= 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen/lot: {gen_per_lot[opt]}, cost/lot: {cost_per_lot[opt]:.2f})")
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')