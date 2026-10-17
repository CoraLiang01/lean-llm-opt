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
options_by_family = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
value_dict = {}
weight_dict = {}
labor_dict = {}
for _, row in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value_dict[fam, opt] = row['Value']
    weight_dict[fam, opt] = row['Weight']
    labor_dict[fam, opt] = row['LaborHours']
limits_map = {r.strip(): l for r, l in zip(limits_df['Resource'], limits_df['Limit'])}
if 'Weight' not in limits_map or 'LaborHours' not in limits_map:
    raise ValueError("Missing required resource limits for 'Weight' or 'LaborHours'.")
weight_limit = limits_map['Weight']
labor_limit = limits_map['LaborHours']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(fam, opt) for fam in families for opt in options_by_family[fam]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in options_by_family[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in options_by_family[fam])) == 1)
m.addConstr(gp.quicksum((weight_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in options_by_family[fam])) <= weight_limit)
m.addConstr(gp.quicksum((labor_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in options_by_family[fam])) <= labor_limit)
m.optimize()