import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
option_ids = list(energy_df['option'])
if energy_df['gen_per_lot'].isnull().any() or energy_df['cost_per_lot'].isnull().any():
    raise ValueError("Missing required numeric coefficients in 'gen_per_lot' or 'cost_per_lot'.")
gen_per_lot = {}
for (idx, row) in energy_df.iterrows():
    option = row['option']
    try:
        gen = int(row['gen_per_lot'])
    except Exception:
        raise ValueError(f"Invalid gen_per_lot for option {option}: {row['gen_per_lot']}")
    gen_per_lot[option] = gen
cost_per_lot = {}
for (idx, row) in energy_df.iterrows():
    option = row['option']
    try:
        cost = float(row['cost_per_lot'])
    except Exception:
        raise ValueError(f"Invalid cost_per_lot for option {option}: {row['cost_per_lot']}")
    cost_per_lot[option] = cost
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option_id coverage for parameters.')
total_demand = 200
m = gp.Model('Electricity_Procurement')
quantity_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * quantity_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * quantity_vars[opt] for opt in option_ids)) >= total_demand, name='demand')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for opt in option_ids:
        var = quantity_vars[opt]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')