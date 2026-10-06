import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
valid_techs = {'coal', 'gas', 'renewables'}
energy_df = energy_df[energy_df['tech'].astype(str).str.casefold().isin(valid_techs)]
options = list(energy_df['option'].astype(str))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
if not options:
    raise ValueError("No valid generation options found for tech in {'coal', 'gas', 'renewables'}.")
for opt in options:
    if opt not in gen_per_lot or opt not in cost_per_lot:
        raise ValueError(f"Missing generation or cost data for option '{opt}'.")
m = gp.Model('ElectricityProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[opt] * x[opt] for opt in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[opt] * x[opt] for opt in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for opt in options:
        val = x[opt].X
        if val >= 1e-06:
            print(f"  Option {opt}: {int(round(val))} lots (tech: {energy_df.loc[energy_df['option'] == opt, 'tech'].values[0]}, gen/lot: {gen_per_lot[opt]}, cost/lot: {cost_per_lot[opt]:.2f})")
    total_gen = sum((gen_per_lot[opt] * x[opt].X for opt in options))
    print(f'Total generation: {total_gen:.2f} (demand: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')