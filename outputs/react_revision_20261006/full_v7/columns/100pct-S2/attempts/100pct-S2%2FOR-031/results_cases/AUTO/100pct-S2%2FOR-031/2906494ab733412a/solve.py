import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = list(energy_df['option'])
if len(option_ids) != len(set(option_ids)):
    raise ValueError("Duplicate 'option' identifiers found in energy.csv")
required_cols = ['gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
try:
    gen_per_lot = pd.Series(energy_df['gen_per_lot'].astype(float).values, index=option_ids)
    cost_per_lot = pd.Series(energy_df['cost_per_lot'].astype(float).values, index=option_ids)
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' or 'cost_per_lot' to float: {e}")
if gen_per_lot.isnull().any():
    raise ValueError("Missing or invalid values in 'gen_per_lot'")
if cost_per_lot.isnull().any():
    raise ValueError("Missing or invalid values in 'cost_per_lot'")
total_demand = 200.0

def solve_generation_lot_purchase():
    m = gp.Model('generation_lot_purchase')
    quantity_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * quantity_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * quantity_vars[i] for i in option_ids)) >= total_demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_generation_lot_purchase()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')