import gurobipy as gp
import pandas as pd
import numpy as np
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
option_ids = list(energy_df['option'])
required_columns = ['option', 'tech', 'gen_per_lot', 'cost_per_lot']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f'Missing required column: {col}')
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for (idx, row) in energy_df.iterrows():
        option = row['option']
        try:
            gen = int(row['gen_per_lot'])
            cost = float(row['cost_per_lot'])
        except Exception as e:
            raise ValueError(f'Invalid numeric value in row {idx} for option {option}: {e}')
        gen_per_lot[option] = gen
        cost_per_lot[option] = cost
except Exception as e:
    raise RuntimeError(f'Error processing parameter data: {e}')
if set(option_ids) != set(gen_per_lot.keys()) or set(option_ids) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch between option IDs and parameter keys.')
total_demand = 200

def solve_generation_lot_selection(option_ids, gen_per_lot, cost_per_lot, total_demand):
    m = gp.Model('Electricity_Generation_Lot_Selection')
    x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in option_ids)) >= total_demand, name='demand')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_generation_lot_selection(option_ids, gen_per_lot, cost_per_lot, total_demand)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal:.6f}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')