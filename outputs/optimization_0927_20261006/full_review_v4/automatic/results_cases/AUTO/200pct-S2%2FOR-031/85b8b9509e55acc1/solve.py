import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',', dtype=str, keep_default_na=False)
eligible_techs = {'coal', 'gas', 'renewables'}
energy_df['tech_norm'] = energy_df['tech'].str.strip().str.casefold()
eligible_df = energy_df[energy_df['tech_norm'].isin({t.casefold() for t in eligible_techs})].copy()
required_columns = ['option', 'gen_per_lot', 'cost_per_lot']
for col in required_columns:
    if col not in eligible_df.columns:
        raise KeyError(f"Required column '{col}' not found in energy.csv")
try:
    eligible_df['gen_per_lot'] = eligible_df['gen_per_lot'].astype(int)
    eligible_df['cost_per_lot'] = eligible_df['cost_per_lot'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting gen_per_lot or cost_per_lot to numeric: {e}')
option_ids = eligible_df['option'].tolist()
gen_per_lot_dict = dict(zip(eligible_df['option'], eligible_df['gen_per_lot']))
cost_per_lot_dict = dict(zip(eligible_df['option'], eligible_df['cost_per_lot']))
m = gp.Model('ElectricityProcurement')
x_vars = m.addVars(option_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot_dict[i] * x_vars[i] for i in option_ids)), gp.GRB.MINIMIZE)
demand = 200
m.addConstr(gp.quicksum((gen_per_lot_dict[i] * x_vars[i] for i in option_ids)) >= demand, name='demand')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Lot purchase plan:')
    for i in option_ids:
        xi = x_vars[i].X
        if xi > 0.5:
            print(f"  Option {i}: {int(round(xi))} lots (tech: {eligible_df.loc[eligible_df['option'] == i, 'tech'].iloc[0]}, gen/lot: {gen_per_lot_dict[i]}, cost/lot: {cost_per_lot_dict[i]:.2f})")
    total_gen = sum((gen_per_lot_dict[i] * x_vars[i].X for i in option_ids))
    print(f'Total generation purchased: {total_gen:.2f} (demand: {demand})')
else:
    print(f'No optimal solution found. Status: {m.status}')