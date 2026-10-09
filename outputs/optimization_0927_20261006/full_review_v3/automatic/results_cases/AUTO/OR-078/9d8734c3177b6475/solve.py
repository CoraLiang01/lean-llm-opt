import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
tech_map = dict(zip(energy_df['option'], energy_df['tech']))
try:
    gen_per_lot_map = {row['option']: int(row['gen_per_lot']) for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to int: {e}")
try:
    cost_per_lot_map = {row['option']: float(row['cost_per_lot']) for (_, row) in energy_df.iterrows()}
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_map[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_map[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='DemandSatisfaction')
m.optimize()