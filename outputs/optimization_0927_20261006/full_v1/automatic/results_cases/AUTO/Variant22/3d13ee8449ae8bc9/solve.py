import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'LaborHours']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
family_options = {}
for fam in families:
    fam_opts = option_df.loc[option_df['Family'] == fam, 'Option'].unique()
    family_options[fam] = sorted(fam_opts)
value = {}
weight = {}
labor = {}
for (idx, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value[fam, opt] = row['Value']
    weight[fam, opt] = row['Weight']
    labor[fam, opt] = row['LaborHours']
resource_limits = {}
for (idx, row) in limits_df.iterrows():
    res = row['Resource'].strip()
    lim = row['Limit']
    resource_limits[res] = lim
for res in ['Weight', 'LaborHours']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(((fam, opt) for fam in families for opt in family_options[fam]), vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'OneOption_{fam}')
m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= resource_limits['Weight'], name='TotalWeight')
m.addConstr(gp.quicksum((labor[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= resource_limits['LaborHours'], name='TotalLabor')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Options ---')
    for fam in families:
        for opt in family_options[fam]:
            if x_vars[fam, opt].X > 0.5:
                print(f'Family {fam}: Option {opt} (Value={value[fam, opt]}, Weight={weight[fam, opt]}, LaborHours={labor[fam, opt]})')
    total_weight = sum((weight[fam, opt] * x_vars[fam, opt].X for fam in families for opt in family_options[fam]))
    total_labor = sum((labor[fam, opt] * x_vars[fam, opt].X for fam in families for opt in family_options[fam]))
    print(f"Total Weight Used: {total_weight:.0f} / {resource_limits['Weight']}")
    print(f"Total LaborHours Used: {total_labor:.0f} / {resource_limits['LaborHours']}")
else:
    print(f'No optimal solution found. Status: {m.status}')