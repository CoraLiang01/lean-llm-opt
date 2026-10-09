import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = energy_df['option'].tolist()
required_cols = ['option', 'tech', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
energy_df = energy_df.set_index('option', drop=False)

def safe_float(x, col, idx):
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Invalid numeric value '{x}' in column '{col}' for option '{idx}'")
gen_per_lot = {}
cost_per_lot = {}
tech = {}
for opt in options:
    row = energy_df.loc[opt]
    gen_per_lot[opt] = safe_float(row['gen_per_lot'], 'gen_per_lot', opt)
    cost_per_lot[opt] = safe_float(row['cost_per_lot'], 'cost_per_lot', opt)
    tech[opt] = row['tech']
m = gp.Model('Electricity_Procurement')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
demand = 200.0
m.addConstr(gp.quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) >= demand, name='DemandSatisfaction')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots by option:')
    for opt in options:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f'  Option: {opt} | Tech: {tech[opt]} | Lots: {int(round(val))} | Gen per lot: {gen_per_lot[opt]} | Cost per lot: {cost_per_lot[opt]}')
    total_gen = sum((gen_per_lot[opt] * x_vars[opt].X for opt in options))
    print(f'Total generation: {total_gen:.2f} (Demand: {demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')