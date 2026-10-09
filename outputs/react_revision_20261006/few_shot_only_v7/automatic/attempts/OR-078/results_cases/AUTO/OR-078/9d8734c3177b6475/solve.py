import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['option', 'tech', 'gen_per_lot', 'cost_per_lot', 'unit_cost_est']
missing_cols = [col for col in required_cols if col not in energy_df.columns]
if missing_cols:
    raise ValueError(f'Missing required columns in energy.csv: {missing_cols}')
option_keys = list(energy_df['option'])

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception:
        raise ValueError(f"Non-numeric value found in column '{colname}' of energy.csv")
gen_per_lot = dict(zip(option_keys, to_float_series(energy_df['gen_per_lot'], 'gen_per_lot')))
cost_per_lot = dict(zip(option_keys, to_float_series(energy_df['cost_per_lot'], 'cost_per_lot')))
if set(gen_per_lot.keys()) != set(option_keys) or set(cost_per_lot.keys()) != set(option_keys):
    raise ValueError('Mismatch in option keys and parameter dictionaries.')
total_demand = 200.0

def solve_problem():
    m = gp.Model('ElectricityProcurement')
    quantity_vars = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * quantity_vars[i] for i in option_keys)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * quantity_vars[i] for i in option_keys)) >= total_demand, name='demand')
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