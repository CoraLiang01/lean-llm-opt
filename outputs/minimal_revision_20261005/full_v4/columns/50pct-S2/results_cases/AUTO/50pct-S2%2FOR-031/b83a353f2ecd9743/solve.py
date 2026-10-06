import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot']))
if set(options) != set(cost_per_lot.keys()) or set(options) != set(gen_per_lot.keys()):
    raise ValueError('Mismatch in options and parameter keys in energy.csv.')
total_demand = 200

def solve_energy_procurement():
    m = gp.Model('energy_procurement')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) == total_demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in options:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_energy_procurement()