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
family_options = {fam: option_df[option_df['Family'] == fam]['Option'].tolist() for fam in families}
value = {}
weight = {}
labor = {}
for _, row in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value[fam, opt] = row['Value']
    weight[fam, opt] = row['Weight']
    labor[fam, opt] = row['LaborHours']
resource_limits = {}
for _, row in limits_df.iterrows():
    res = row['Resource'].strip()
    lim = row['Limit']
    resource_limits[res.casefold()] = lim
if 'weight' not in resource_limits or 'laborhours' not in resource_limits:
    raise ValueError("Missing required resource limits for 'Weight' and/or 'LaborHours'.")
weight_limit = resource_limits['weight']
labor_limit = resource_limits['laborhours']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(fam, opt) for fam in families for opt in family_options[fam]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1)
m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= weight_limit)
m.addConstr(gp.quicksum((labor[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= labor_limit)
m.optimize()