import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
tech_dict = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
gen_per_lot_dict = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(int)))
cost_per_lot_dict = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
if len(options) != len(set(options)):
    raise ValueError('Duplicate option identifiers found in energy.csv.')
if not all((opt in gen_per_lot_dict and opt in cost_per_lot_dict for opt in options)):
    raise ValueError('Missing gen_per_lot or cost_per_lot for some options.')
m = gp.Model('Electricity_Lot_Procurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot_dict[opt] * x[opt] for opt in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Selected lots per option (nonzero only):')
    for opt in options:
        val = x[opt].X
        if val > 1e-06:
            print(f'  Option: {opt:15s} | Tech: {tech_dict[opt]:11s} | Lots: {int(round(val))} | Gen/lot: {gen_per_lot_dict[opt]} | Cost/lot: {cost_per_lot_dict[opt]:.2f}')
    total_gen = sum((gen_per_lot_dict[opt] * x[opt].X for opt in options))
    print(f'Total generation: {total_gen:.2f} (Demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')