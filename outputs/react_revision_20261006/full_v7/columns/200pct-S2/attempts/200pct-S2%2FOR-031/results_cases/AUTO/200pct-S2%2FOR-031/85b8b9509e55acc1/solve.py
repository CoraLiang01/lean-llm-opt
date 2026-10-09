import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
valid_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
selected_df = energy_df[energy_df['tech_norm'].isin(valid_techs)].copy()
required_cols = ['option', 'cost_per_lot', 'gen_per_lot']
for col in required_cols:
    if col not in selected_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
option_keys = selected_df['option'].tolist()
if len(option_keys) == 0:
    raise ValueError("No generation options found for tech in {'coal', 'gas', 'renewables'}.")
cost_per_lot = {}
gen_per_lot = {}
for (idx, row) in selected_df.iterrows():
    option = row['option']
    try:
        cost = float(row['cost_per_lot'])
        gen = int(row['gen_per_lot'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in row for option '{option}': {e}")
    cost_per_lot[option] = cost
    gen_per_lot[option] = gen
if set(option_keys) != set(cost_per_lot.keys()) or set(option_keys) != set(gen_per_lot.keys()):
    raise ValueError('Mismatch in parameter coverage for options.')
total_demand = 200
m = gp.Model('ElectricityLotProcurement')
quantity_vars = m.addVars(option_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * quantity_vars[opt] for opt in option_keys)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * quantity_vars[opt] for opt in option_keys)) >= total_demand, name='demand')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for opt in option_keys:
        print(f'{quantity_vars[opt].VarName} {quantity_vars[opt].X}')
else:
    print(f'Solver status: {m.Status}')