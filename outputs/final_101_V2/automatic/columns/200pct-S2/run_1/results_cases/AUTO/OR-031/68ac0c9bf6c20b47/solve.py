import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
valid_techs = {'coal', 'gas', 'renewables'}
energy_df['tech'] = energy_df['tech'].astype(str).str.casefold().str.strip()
filtered_df = energy_df[energy_df['tech'].isin(valid_techs)].copy()
options = filtered_df['option'].astype(str).tolist()
if filtered_df[['option', 'gen_per_lot', 'cost_per_lot']].isnull().any().any():
    raise ValueError("Missing required data in 'option', 'gen_per_lot', or 'cost_per_lot' columns.")
gen_per_lot = dict(zip(filtered_df['option'].astype(str), filtered_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(filtered_df['option'].astype(str), filtered_df['cost_per_lot'].astype(float)))
for i in options:
    if i not in gen_per_lot or i not in cost_per_lot:
        raise KeyError(f"Missing coefficients for option '{i}'.")
m = gp.Model('ElectricityLotProcurement')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
total_demand = 200
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='Demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in options:
        xi = x[i].X
        if xi >= 1e-06:
            print(f'  {i}: {int(round(xi))} lots (gen_per_lot={gen_per_lot[i]}, cost_per_lot={cost_per_lot[i]:.2f})')
    total_generation = sum((gen_per_lot[i] * x[i].X for i in options))
    print(f'Total generation: {total_generation:.2f} (demand required: {total_demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')