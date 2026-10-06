import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
cost_per_lot = {}
gen_per_lot = {}
tech = {}
for idx, row in energy_df.iterrows():
    opt = str(row['option'])
    cost_per_lot[opt] = float(row['cost_per_lot'])
    gen_per_lot[opt] = int(row['gen_per_lot'])
    tech[opt] = str(row['tech'])
m = gp.Model('ElectricityProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x[opt] for opt in options)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for opt in options:
        val = x[opt].X
        if val >= 1e-06:
            print(f'  Option {opt} (tech={tech[opt]}): {int(round(val))} lots, Total gen: {gen_per_lot[opt] * int(round(val))}, Total cost: {cost_per_lot[opt] * int(round(val)):.2f}')
    total_gen = sum((gen_per_lot[opt] * x[opt].X for opt in options))
    print(f'Total generation: {total_gen:.2f} (required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')