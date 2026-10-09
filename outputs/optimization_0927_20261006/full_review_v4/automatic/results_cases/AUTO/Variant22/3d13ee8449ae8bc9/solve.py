import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'LaborHours']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
fam_opt_pairs = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value_dict = {(row['Family'], row['Option']): row['Value'] for (_, row) in option_df.iterrows()}
weight_dict = {(row['Family'], row['Option']): row['Weight'] for (_, row) in option_df.iterrows()}
labor_dict = {(row['Family'], row['Option']): row['LaborHours'] for (_, row) in option_df.iterrows()}
limits_map = {row['Resource'].strip().casefold(): row['Limit'] for (_, row) in limits_df.iterrows()}
for res in ['weight', 'laborhours']:
    if res not in limits_map:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
weight_limit = limits_map['weight']
labor_limit = limits_map['laborhours']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[fam_opt] * x_vars[fam_opt] for fam_opt in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'OneOption_{fam}')
m.addConstr(gp.quicksum((weight_dict[fam_opt] * x_vars[fam_opt] for fam_opt in fam_opt_pairs)) <= weight_limit, name='TotalWeight')
m.addConstr(gp.quicksum((labor_dict[fam_opt] * x_vars[fam_opt] for fam_opt in fam_opt_pairs)) <= labor_limit, name='TotalLaborHours')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Options ---')
    for fam in families:
        for opt in family_options[fam]:
            if x_vars[fam, opt].X > 0.5:
                val = value_dict[fam, opt]
                wt = weight_dict[fam, opt]
                lh = labor_dict[fam, opt]
                print(f'Family {fam}: Option {opt} (Value={val}, Weight={wt}, LaborHours={lh})')
else:
    print(f'No optimal solution found. Status: {m.status}')