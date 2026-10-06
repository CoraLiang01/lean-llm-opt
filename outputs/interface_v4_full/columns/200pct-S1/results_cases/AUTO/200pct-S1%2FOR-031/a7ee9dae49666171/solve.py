import gurobipy as gp
import pandas as pd
import numpy as np
energy_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture5/energy.csv'
energy_df = pd.read_csv(energy_path, sep=',')
energy_df['tech_norm'] = energy_df['tech'].astype(str).str.strip().str.casefold()
allowed_techs = {'coal', 'gas', 'renewables'}
allowed_techs_norm = {t.casefold() for t in allowed_techs}
filtered_df = energy_df[energy_df['tech_norm'].isin(allowed_techs_norm)].copy()
required_cols = ['option', 'gen_per_lot', 'cost_per_lot', 'tech']
for col in required_cols:
    if col not in filtered_df.columns:
        raise KeyError(f"Missing required column '{col}' in energy.csv")
options = filtered_df['option'].astype(str).tolist()
gen_per_lot = filtered_df.set_index('option')['gen_per_lot'].astype(float).to_dict()
cost_per_lot = filtered_df.set_index('option')['cost_per_lot'].astype(float).to_dict()
tech = filtered_df.set_index('option')['tech'].astype(str).to_dict()
for i in options:
    if i not in gen_per_lot or i not in cost_per_lot or i not in tech:
        raise ValueError(f"Missing parameter for option '{i}'")
total_demand = 200.0
m = gp.Model('ElectricityProcurement_MIP')
x = m.addVars(options, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((cost_per_lot[i] * x[i] for i in options)), gp.GRB.MINIMIZE)
m.addConstr(gp.quicksum((gen_per_lot[i] * x[i] for i in options)) >= total_demand, name='DemandSatisfaction')
m.optimize()