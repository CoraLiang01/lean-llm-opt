import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
energy_df['option_norm'] = energy_df['option'].astype(str).str.strip()
energy_df['gen_per_lot'] = energy_df['gen_per_lot'].astype(float)
energy_df['cost_per_lot'] = energy_df['cost_per_lot'].astype(float)
energy_df['tech_norm'] = energy_df['tech'].astype(str).str.strip().str.casefold()
gen_per_lot = dict(zip(energy_df['option_norm'], energy_df['gen_per_lot']))
cost_per_lot = dict(zip(energy_df['option_norm'], energy_df['cost_per_lot']))
tech = dict(zip(energy_df['option_norm'], energy_df['tech_norm']))
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='demand')
m.optimize()