import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'])
cost_per_lot = dict(zip(energy_df['option'], energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'], energy_df['gen_per_lot']))
if set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in cost_per_lot keys and options.')
if set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in gen_per_lot keys and options.')
total_demand = 200

def solve_generation_procurement(options, cost_per_lot, gen_per_lot, total_demand):
    m = gp.Model('generation_procurement')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
    m.optimize()
    return m
m = solve_generation_procurement(options, cost_per_lot, gen_per_lot, total_demand)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')