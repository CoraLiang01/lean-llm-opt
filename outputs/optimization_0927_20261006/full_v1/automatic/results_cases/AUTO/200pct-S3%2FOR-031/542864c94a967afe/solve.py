import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
valid_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
selected_rows = energy_df['tech_norm'].isin(valid_techs)
filtered_df = energy_df[selected_rows].copy()
option_ids = filtered_df['option'].tolist()
for col in ['cost_per_lot', 'gen_per_lot']:
    if filtered_df[col].isnull().any() or (filtered_df[col].str.strip() == '').any():
        raise ValueError(f"Missing or blank values in required column '{col}' for selected options.")
filtered_df['cost_per_lot_num'] = filtered_df['cost_per_lot'].astype(float)
filtered_df['gen_per_lot_num'] = filtered_df['gen_per_lot'].astype(float)
cost_per_lot = dict(zip(filtered_df['option'], filtered_df['cost_per_lot_num']))
gen_per_lot = dict(zip(filtered_df['option'], filtered_df['gen_per_lot_num']))
m = gp.Model('ElectricityLotProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x_vars[i] for i in option_ids)) >= 200, name='demand')
m.optimize()