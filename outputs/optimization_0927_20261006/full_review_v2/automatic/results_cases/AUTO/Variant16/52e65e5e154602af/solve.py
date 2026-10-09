import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
option_df['Family'] = option_df['Family'].str.strip()
option_df['Option'] = option_df['Option'].str.strip()
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Resource'] = limits_df['Resource'].str.strip()
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
fam_opt_pairs = []
for fam in families:
    for opt in family_options[fam]:
        fam_opt_pairs.append((fam, opt))
value = {}
weight = {}
budget_use = {}
for (idx, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value[fam, opt] = int(row['Value'])
    weight[fam, opt] = int(row['Weight'])
    budget_use[fam, opt] = int(row['BudgetUse'])

def get_limit(resource_name):
    norm = resource_name.strip().casefold()
    match = limits_df['Resource'].str.strip().str.casefold() == norm
    if not match.any():
        raise ValueError(f"Resource '{resource_name}' not found in resource_limits.csv")
    return int(limits_df.loc[match, 'Limit'].iloc[0])
weight_limit = get_limit('Weight')
budget_limit = get_limit('BudgetUse')
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1)
m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)) <= weight_limit)
m.addConstr(gp.quicksum((budget_use[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)) <= budget_limit)
m.optimize()