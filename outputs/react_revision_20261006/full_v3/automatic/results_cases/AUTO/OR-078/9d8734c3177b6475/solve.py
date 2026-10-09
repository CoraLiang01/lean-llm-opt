import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
option_ids = list(energy_df['option'].astype(str))
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if energy_df[col].isnull().any():
        raise ValueError(f"Missing values found in required column '{col}'.")
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option IDs and parameter keys.')
total_demand = 200.0
m = gp.Model('Electricity_Procurement_MIP')
x = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in option_ids)) >= total_demand, name='demand')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal:.6f}')
    for i in option_ids:
        var = x[i]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')