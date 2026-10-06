import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
if 'option' not in energy_df.columns or 'gen_per_lot' not in energy_df.columns or 'cost_per_lot' not in energy_df.columns:
    raise KeyError('Missing required columns in energy.csv')
energy_df['option'] = energy_df['option'].astype(str)
options = list(energy_df['option'].unique())
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in option keys and parameter dictionaries.')

def solve_problem(options, gen_per_lot, cost_per_lot):
    m = gp.Model('ElectricityProcurement')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(options, gen_per_lot, cost_per_lot)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')