import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
options = list(energy_df['option'].astype(str))
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
tech = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
if set(gen_per_lot.keys()) != set(options) or set(cost_per_lot.keys()) != set(options):
    raise ValueError('Mismatch in parameter keys and options in energy.csv.')
m = gp.Model('Electricity_Procurement_Lot_Selection')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Selected lots by option:')
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f'  Option: {i:15s} | Tech: {tech[i]:11s} | Lots: {int(round(xi))} | Gen per lot: {gen_per_lot[i]:.2f} | Cost per lot: {cost_per_lot[i]:.2f}')
    total_gen = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation: {total_gen:.2f} (Demand: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')