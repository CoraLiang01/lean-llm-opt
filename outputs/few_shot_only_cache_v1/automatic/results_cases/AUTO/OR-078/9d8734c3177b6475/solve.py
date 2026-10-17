import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = energy_df['option'].tolist()
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
try:
    gen_per_lot = {}
    cost_per_lot = {}
    for idx, row in energy_df.iterrows():
        option = row['option']
        try:
            gen_per_lot[option] = float(row['gen_per_lot'])
        except Exception:
            raise ValueError(f"Invalid gen_per_lot value for option '{option}': {row['gen_per_lot']}")
        try:
            cost_per_lot[option] = float(row['cost_per_lot'])
        except Exception:
            raise ValueError(f"Invalid cost_per_lot value for option '{option}': {row['cost_per_lot']}")
except Exception as e:
    raise RuntimeError(f'Error processing energy.csv: {e}')
m = gp.Model('Electricity_Lot_Scheduling')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[o] * x_vars[o] for o in options)), gp.GRB.MINIMIZE)
total_demand = 200.0
m.addConstr(gp.quicksum((gen_per_lot[o] * x_vars[o] for o in options)) >= total_demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.4f}')
    print('Selected lots per option (nonzero only):')
    for o in options:
        val = x_vars[o].X
        if val > 1e-06:
            print(f'  {o}: {int(round(val))} lots (gen_per_lot={gen_per_lot[o]}, cost_per_lot={cost_per_lot[o]})')
    total_gen = sum((gen_per_lot[o] * x_vars[o].X for o in options))
    print(f'Total generation: {total_gen:.4f} (demand required: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')