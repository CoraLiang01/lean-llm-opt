import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to float: {e}")
try:
    cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
tech = energy_df.set_index('option')['tech'].to_dict()
for oid in option_ids:
    if oid not in gen_per_lot or oid not in cost_per_lot or oid not in tech:
        raise KeyError(f"Missing required data for option '{oid}'.")
m = gp.Model('ElectricityLotProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[oid] * x_vars[oid] for oid in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[oid] * x_vars[oid] for oid in option_ids)) >= 200, name='DemandSatisfaction')
m.optimize()