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
options = sorted(option_df['Option'].unique())
family_options = {}
for fam in families:
    family_options[fam] = sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique())
famopt_keys = []
for fam in families:
    fam_opts = family_options[fam]
    for opt in fam_opts:
        famopt_keys.append((fam, opt))
value = {}
weight = {}
laborhours = {}
for (idx, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    key = (fam, opt)
    value[key] = int(row['Value'])
    weight[key] = int(row['Weight'])
    laborhours[key] = int(row['LaborHours'])
limits_dict = {}
for (idx, row) in limits_df.iterrows():
    res = row['Resource'].strip()
    lim = int(row['Limit'])
    limits_dict[res] = lim
for res in ['Weight', 'LaborHours']:
    if res not in limits_dict:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('multi_choice_knapsack')
x_vars = m.addVars(famopt_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[key] * x_vars[key] for key in famopt_keys)), gp.GRB.MAXIMIZE)
for fam in families:
    fam_keys = [(fam, opt) for opt in family_options[fam]]
    m.addConstr(gp.quicksum((x_vars[key] for key in fam_keys)) == 1, name='oneopt_' + fam)
m.addConstr(gp.quicksum((weight[key] * x_vars[key] for key in famopt_keys)) <= limits_dict['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((laborhours[key] * x_vars[key] for key in famopt_keys)) <= limits_dict['LaborHours'], name='laborhour_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for key in famopt_keys:
        var = x_vars[key]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.Status}')