import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(int)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
tech = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
if set(gen_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in gen_per_lot keys and options.')
if set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in cost_per_lot keys and options.')
m = gp.Model('ElectricityProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    tech_groups = {}
    for i in options:
        t = tech[i]
        if t not in tech_groups:
            tech_groups[t] = []
        tech_groups[t].append(i)
    for t in sorted(tech_groups):
        print(f'  {t.capitalize()}:')
        for i in tech_groups[t]:
            xi = x[i].X
            if xi >= 1e-06:
                print(f'    Option {i}: {int(round(xi))} lots (gen/lot={gen_per_lot[i]}, cost/lot={cost_per_lot[i]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')