import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
    df = pd.read_csv(energy_path, sep=',')
    options = df['option'].astype(str).tolist()
    n_options = len(options)
    cost_per_lot = df.set_index('option')['cost_per_lot'].to_dict()
    gen_per_lot = df.set_index('option')['gen_per_lot'].to_dict()
    if set(options) != set(cost_per_lot.keys()):
        raise ValueError('Mismatch between options and cost_per_lot keys')
    if set(options) != set(gen_per_lot.keys()):
        raise ValueError('Mismatch between options and gen_per_lot keys')
    m = gp.Model('ElectricityProcurement')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in options:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()