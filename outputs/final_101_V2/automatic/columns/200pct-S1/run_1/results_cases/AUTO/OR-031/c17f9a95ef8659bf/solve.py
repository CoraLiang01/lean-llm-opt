import gurobipy as gp
import pandas as pd
import numpy as np
import re
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
lot_ids = energy_df['option'].astype(str).tolist()
required_cols = ['option', 'gen_per_lot', 'cost_per_lot', 'tech']
for col in required_cols:
    if col not in energy_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
gen_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['gen_per_lot'].astype(float)))
cost_per_lot = dict(zip(energy_df['option'].astype(str), energy_df['cost_per_lot'].astype(float)))
tech_of_lot = dict(zip(energy_df['option'].astype(str), energy_df['tech'].astype(str)))
m = gp.Model('ElectricityProcurementLots')
x = m.addVars(lot_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in lot_ids)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in lot_ids)) >= 200, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total procurement cost: {m.objVal:.2f}')
    print('Lot purchase plan (nonzero lots):')
    for i in lot_ids:
        xi = x[i].X
        if xi > 1e-06:
            print(f'  Lot {i} (tech={tech_of_lot[i]}): {int(round(xi))} lot(s), gen_per_lot={gen_per_lot[i]}, cost_per_lot={cost_per_lot[i]:.2f}')
    total_gen = sum((gen_per_lot[i] * x[i].X for i in lot_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand required: 200)')
else:
    print(f'No optimal solution found. Status: {m.status}')