import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
if not set(['option', 'gen_per_lot', 'cost_per_lot']).issubset(energy_df.columns):
    raise ValueError('Missing required columns in energy.csv')
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option keys between data and parameter dictionaries.')
total_demand = 200.0
m = gp.Model('Electricity_Procurement_Lots')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for i in options:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')