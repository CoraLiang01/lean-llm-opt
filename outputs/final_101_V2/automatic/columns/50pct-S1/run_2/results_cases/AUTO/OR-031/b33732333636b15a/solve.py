import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(int)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
if set(gen_per_lot.keys()) != set(options) or set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in option keys for coefficients.')
m = gp.Model('Electricity_Lot_Purchasing')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in options:
        xi = int(round(x[i].X))
        if xi > 0:
            print(f"  Option {i}: {xi} lot(s) (tech={energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen/lot={gen_per_lot[i]}, cost/lot={cost_per_lot[i]:.2f})")
    total_gen = sum((gen_per_lot[i] * int(round(x[i].X)) for i in options))
    print(f'Total generation: {total_gen}')
else:
    print(f'No optimal solution found. Status: {m.status}')