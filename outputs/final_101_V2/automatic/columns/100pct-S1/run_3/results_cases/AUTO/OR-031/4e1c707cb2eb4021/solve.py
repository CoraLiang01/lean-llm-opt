import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'].astype(str))
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
tech = energy_df.set_index('option')['tech'].astype(str).to_dict()
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in options and parameter keys in energy.csv.')
m = gp.Model('ElectricityLotProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Selected lots per option:')
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f'  Option {i} (tech={tech[i]}): {int(round(xi))} lots, Total gen={gen_per_lot[i] * int(round(xi))}, Total cost={cost_per_lot[i] * int(round(xi)):.2f}')
    total_gen = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')