import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'].astype(str).unique())
cost_per_lot = {}
gen_per_lot = {}
if energy_df['cost_per_lot'].isnull().any():
    raise ValueError("Missing values found in 'cost_per_lot' column.")
if energy_df['gen_per_lot'].isnull().any():
    raise ValueError("Missing values found in 'gen_per_lot' column.")
for (_, row) in energy_df.iterrows():
    option = str(row['option'])
    cost = float(row['cost_per_lot'])
    gen = float(row['gen_per_lot'])
    cost_per_lot[option] = cost
    gen_per_lot[option] = gen
if set(cost_per_lot.keys()) != set(options) or set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in parameter keys and option set.')
total_demand = 200.0

def solve_problem():
    m = gp.Model('Electricity_Lot_Purchasing')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')