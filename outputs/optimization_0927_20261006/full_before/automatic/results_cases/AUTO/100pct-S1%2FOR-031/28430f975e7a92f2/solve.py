import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
if 'option' not in energy_df.columns:
    raise KeyError("Missing required column 'option' in energy.csv")
options = list(energy_df['option'].astype(str))
required_cols = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
m = gp.Model('Electricity_Procurement_Lot_Selection')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected contract lots (option, tech, lots, gen_per_lot, cost_per_lot):')
    tech_dict = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f'  {i:15s}  {tech_dict[i]:12s}  {int(round(xi)):3d}  {gen_per_lot[i]:6.1f}  {cost_per_lot[i]:8.2f}')
    total_gen = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')