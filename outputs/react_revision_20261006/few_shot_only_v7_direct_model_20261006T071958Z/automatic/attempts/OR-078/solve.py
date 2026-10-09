import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = list(energy_df['option'])
required_cols = ['gen_per_lot', 'cost_per_lot', 'tech']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values: {e}")
gen_per_lot = dict(zip(energy_df['option'], to_float_series(energy_df['gen_per_lot'], 'gen_per_lot')))
cost_per_lot = dict(zip(energy_df['option'], to_float_series(energy_df['cost_per_lot'], 'cost_per_lot')))
tech = dict(zip(energy_df['option'], energy_df['tech'].astype(str)))
for i in options:
    if i not in gen_per_lot or i not in cost_per_lot or i not in tech:
        raise ValueError(f"Missing parameter for option '{i}'")
total_demand = 200.0

def solve_energy_lot_sizing():
    m = gp.Model('energy_lot_sizing')
    quantity_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((cost_per_lot[i] * quantity_vars[i] for i in options)), gp.GRB.MINIMIZE)
    m.addConstr(gp.quicksum((gen_per_lot[i] * quantity_vars[i] for i in options)) == total_demand, name='demand')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_energy_lot_sizing()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')