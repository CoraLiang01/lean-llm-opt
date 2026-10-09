import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
option_ids = energy_df['option'].tolist()
required_columns = ['gen_per_lot', 'cost_per_lot', 'option']
for col in required_columns:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
try:
    gen_per_lot = pd.to_numeric(energy_df['gen_per_lot'], errors='raise')
    cost_per_lot = pd.to_numeric(energy_df['cost_per_lot'], errors='raise')
except Exception as e:
    raise ValueError(f'Error converting gen_per_lot or cost_per_lot to numeric: {e}')
gen_per_lot_dict = dict(zip(energy_df['option'], gen_per_lot))
cost_per_lot_dict = dict(zip(energy_df['option'], cost_per_lot))
m = gp.Model('Electricity_Lot_Purchasing')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x_vars[opt] for opt in option_ids)) >= 200, name='TotalDemand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for opt in option_ids:
        val = x_vars[opt].X
        if val > 0.5:
            print(f"  {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen_per_lot: {gen_per_lot_dict[opt]}, cost_per_lot: {cost_per_lot_dict[opt]:.2f})")
    total_gen = sum((gen_per_lot_dict[opt] * x_vars[opt].X for opt in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')