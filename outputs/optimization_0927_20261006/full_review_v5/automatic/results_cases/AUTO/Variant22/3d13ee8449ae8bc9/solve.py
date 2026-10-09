import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
option_df['Family'] = option_df['Family'].str.strip()
option_df['Option'] = option_df['Option'].str.strip()
for col in ['Value', 'Weight', 'LaborHours']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Resource'] = limits_df['Resource'].str.strip()
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
value_param = {}
weight_param = {}
labor_param = {}
for (idx, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value_param[fam, opt] = row['Value']
    weight_param[fam, opt] = row['Weight']
    labor_param[fam, opt] = row['LaborHours']
resource_limits = {}
for (idx, row) in limits_df.iterrows():
    res = row['Resource'].strip()
    resource_limits[res] = row['Limit']
required_resources = ['Weight', 'LaborHours']
for res in required_resources:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
weight_limit = resource_limits['Weight']
labor_limit = resource_limits['LaborHours']
m = gp.Model('MultiChoiceKnapsack')
x_vars = {}
for fam in families:
    for opt in family_options[fam]:
        x_vars[fam, opt] = m.addVar(vtype=gp.GRB.BINARY, name=f'x_{fam}_{opt}')
m.setObjective(gp.quicksum((value_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'OneOption_{fam}')
m.addConstr(gp.quicksum((weight_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= weight_limit, name='TotalWeight')
m.addConstr(gp.quicksum((labor_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= labor_limit, name='TotalLaborHours')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Options ---')
    for fam in families:
        for opt in family_options[fam]:
            if x_vars[fam, opt].X > 0.5:
                print(f'Family {fam}: Option {opt} (Value={value_param[fam, opt]}, Weight={weight_param[fam, opt]}, LaborHours={labor_param[fam, opt]})')
    total_weight = sum((weight_param[fam, opt] * x_vars[fam, opt].X for fam in families for opt in family_options[fam]))
    total_labor = sum((labor_param[fam, opt] * x_vars[fam, opt].X for fam in families for opt in family_options[fam]))
    print(f'Total Weight Used: {total_weight} / {weight_limit}')
    print(f'Total LaborHours Used: {total_labor} / {labor_limit}')
else:
    print(f'No optimal solution found. Status: {m.status}')