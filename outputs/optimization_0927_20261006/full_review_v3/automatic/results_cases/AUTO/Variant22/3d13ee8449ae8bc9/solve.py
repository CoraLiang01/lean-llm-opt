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
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
value_param = {}
weight_param = {}
labor_param = {}
for (_, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value_param[fam, opt] = row['Value']
    weight_param[fam, opt] = row['Weight']
    labor_param[fam, opt] = row['LaborHours']
limits_dict = {row['Resource'].strip(): row['Limit'] for (_, row) in limits_df.iterrows()}
if 'Weight' not in limits_dict or 'LaborHours' not in limits_dict:
    raise ValueError("Resource limits for 'Weight' and/or 'LaborHours' not found in resource_limits.csv")
weight_limit = limits_dict['Weight']
labor_limit = limits_dict['LaborHours']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(fam, opt) for fam in families for opt in family_options[fam]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'OneOption_{fam}')
m.addConstr(gp.quicksum((weight_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= weight_limit, name='TotalWeight')
m.addConstr(gp.quicksum((labor_param[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= labor_limit, name='TotalLaborHours')
m.optimize()