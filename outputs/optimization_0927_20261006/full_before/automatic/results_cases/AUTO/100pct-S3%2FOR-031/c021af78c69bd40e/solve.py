import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(int).to_dict()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
if set(options) != set(gen_per_lot.keys()) or set(options) != set(cost_per_lot.keys()):
    raise ValueError('Mismatch in options and parameter keys in energy.csv.')
m = gp.Model('ElectricityProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f"  Option {i}: {int(round(xi))} lots (tech: {energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen/lot: {gen_per_lot[i]}, cost/lot: {cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')