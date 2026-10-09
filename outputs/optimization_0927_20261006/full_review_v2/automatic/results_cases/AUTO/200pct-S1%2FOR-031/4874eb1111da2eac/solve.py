import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Column matching '{pattern}' not found.")
option_col = 'option'
tech_col = 'tech'
gen_per_lot_col = 'gen_per_lot'
cost_per_lot_col = 'cost_per_lot'
options = list(energy_df[option_col])
option_to_tech = dict(zip(energy_df[option_col], energy_df[tech_col]))
try:
    gen_per_lot_series = energy_df.set_index(option_col)[gen_per_lot_col].apply(lambda x: int(x.strip()))
except Exception as e:
    raise ValueError(f"Error converting 'gen_per_lot' to int: {e}")
option_to_gen_per_lot = gen_per_lot_series.to_dict()
try:
    cost_per_lot_series = energy_df.set_index(option_col)[cost_per_lot_col].apply(lambda x: float(x.strip()))
except Exception as e:
    raise ValueError(f"Error converting 'cost_per_lot' to float: {e}")
option_to_cost_per_lot = cost_per_lot_series.to_dict()
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((option_to_cost_per_lot[opt] * x_vars[opt] for opt in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((option_to_gen_per_lot[opt] * x_vars[opt] for opt in options)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for opt in options:
        val = x_vars[opt].X
        if val > 1e-06:
            print(f'  Option: {opt} | Tech: {option_to_tech[opt]} | Lots: {int(round(val))} | Gen/lot: {option_to_gen_per_lot[opt]} | Cost/lot: {option_to_cost_per_lot[opt]:.2f}')
    total_gen = sum((option_to_gen_per_lot[opt] * x_vars[opt].X for opt in options))
    print(f'Total generation: {total_gen:.2f} (Demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')