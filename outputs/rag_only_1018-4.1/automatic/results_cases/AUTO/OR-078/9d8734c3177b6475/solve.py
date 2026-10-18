import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
energy_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv', dtype=str, keep_default_na=False)
required_columns = ['option', 'tech', 'gen_per_lot', 'cost_per_lot', 'unit_cost_est']
for col in required_columns:
    if col not in energy_df.columns:
        raise ValueError(f'Missing required column: {col}')
energy_df['option'] = energy_df['option'].astype(str)
energy_df = energy_df.set_index('option', drop=False)
for col in ['gen_per_lot', 'cost_per_lot', 'unit_cost_est']:
    try:
        energy_df[col] = pd.to_numeric(energy_df[col], errors='raise')
    except Exception as e:
        raise ValueError(f'Error converting column {col} to numeric: {e}')
options = list(energy_df.index)
gen_per_lot = energy_df['gen_per_lot'].to_dict()
cost_per_lot = energy_df['cost_per_lot'].to_dict()
total_demand = 200
m = Model('electricity_lot_sizing')
x_vars = m.addVars(options, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((cost_per_lot[opt] * x_vars[opt] for opt in options)), GRB.MINIMIZE)
m.addConstr(quicksum((gen_per_lot[opt] * x_vars[opt] for opt in options)) >= total_demand, name='demand')
m.optimize()