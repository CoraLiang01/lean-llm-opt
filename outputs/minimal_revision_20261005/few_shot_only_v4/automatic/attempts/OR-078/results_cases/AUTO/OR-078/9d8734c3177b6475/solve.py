import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'])
required_cols = ['option', 'tech', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
    if energy_df[col].isnull().any():
        raise ValueError(f"Column '{col}' contains missing values in energy.csv")
gen_per_lot = {}
cost_per_lot = {}
for (_, row) in energy_df.iterrows():
    key = row['option']
    gen_per_lot[key] = float(row['gen_per_lot'])
    cost_per_lot[key] = float(row['cost_per_lot'])
if set(gen_per_lot.keys()) != set(options) or set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in parameter keys and options in energy.csv')
demand = 200.0

def solve_generation_lot_problem():
    m = gp.Model('generation_lot_selection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= demand, name='demand')
    m.optimize()
    return m
m = solve_generation_lot_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')