import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = energy_df['option'].astype(str).tolist()
gen_per_lot = energy_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = energy_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
m = gp.Model('Electricity_Lot_Purchasing')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) == 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in options:
        xi = x[i].X
        if xi > 1e-06:
            print(f"  Option {i}: {int(round(xi))} lots (tech: {energy_df.loc[energy_df['option'] == i, 'tech'].values[0]}, gen_per_lot: {gen_per_lot[i]}, cost_per_lot: {cost_per_lot[i]})")
else:
    print(f'No optimal solution found. Status: {m.status}')