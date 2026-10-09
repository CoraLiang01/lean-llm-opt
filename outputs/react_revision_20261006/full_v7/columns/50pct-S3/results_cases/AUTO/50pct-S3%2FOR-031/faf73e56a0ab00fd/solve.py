import gurobipy as gp
import pandas as pd
import numpy as np
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
option_ids = list(energy_df['option'])
required_columns = ['option', 'tech', 'cost_per_lot', 'gen_per_lot']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f'Missing required column: {col}')
cost_per_lot = {}
gen_per_lot = {}
for (idx, row) in energy_df.iterrows():
    option = row['option']
    try:
        cost = float(row['cost_per_lot'])
    except Exception:
        raise ValueError(f"Invalid cost_per_lot for option {option}: {row['cost_per_lot']}")
    try:
        gen = int(row['gen_per_lot'])
    except Exception:
        raise ValueError(f"Invalid gen_per_lot for option {option}: {row['gen_per_lot']}")
    cost_per_lot[option] = cost
    gen_per_lot[option] = gen
if set(option_ids) != set(cost_per_lot.keys()) or set(option_ids) != set(gen_per_lot.keys()):
    raise ValueError('Mismatch in option identifiers between data and parameter dictionaries.')
total_demand = 200
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='demand')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')