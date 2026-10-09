import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = energy_df['option'].tolist()
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        try:
            gen = int(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Invalid gen_per_lot for option '{option}': {row['gen_per_lot']}")
        try:
            cost = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Invalid cost_per_lot for option '{option}': {row['cost_per_lot']}")
        gen_per_lot[option] = gen
        cost_per_lot[option] = cost
except KeyError as e:
    raise KeyError(f'Missing required column in energy.csv: {e}')
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option IDs and parameter keys in energy.csv.')
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[option] * x_vars[option] for option in option_ids)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[option] * x_vars[option] for option in option_ids)) >= total_demand, name='DemandSatisfaction')
m.optimize()