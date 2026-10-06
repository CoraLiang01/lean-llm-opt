import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
option_ids = list(energy_df['option'].astype(str))
required_cols = ['option', 'tech', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
    if energy_df[col].isnull().any():
        raise ValueError(f"Missing values found in column '{col}' of energy.csv")
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
tech = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option_id keys between index set and parameter dictionaries.')
demand = 200.0

def solve_problem():
    m = gp.Model('Electricity_Procurement_MIP')
    m.Params.MIPGap = 0.0001
    x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, obj=0.0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in option_ids)) >= demand, name='demand')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')