import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
if energy_df['option'].duplicated().any():
    raise ValueError("Duplicate 'option' identifiers found in energy.csv.")
options = list(energy_df['option'].astype(str))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot']))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot']))
if set(cost_per_lot.keys()) != set(options) or set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in parameter coverage for options.')
total_demand = 200

def solve_problem():
    m = gp.Model('Electricity_Procurement_Lot_MIP')
    x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')