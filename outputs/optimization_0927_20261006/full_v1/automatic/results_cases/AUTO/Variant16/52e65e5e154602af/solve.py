import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options_by_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
fam_opt_pairs = []
for fam in families:
    fam_opt_pairs.extend([(fam, opt) for opt in options_by_family[fam]])
value = {}
weight = {}
budget_use = {}
for (_, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value[fam, opt] = int(row['Value'])
    weight[fam, opt] = int(row['Weight'])
    budget_use[fam, opt] = int(row['BudgetUse'])
resource_limits = {}
for (_, row) in limits_df.iterrows():
    resource = row['Resource'].strip()
    resource_limits[resource] = int(row['Limit'])
for res in ['Weight', 'BudgetUse']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('multi_choice_knapsack')
x_vars = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in options_by_family[fam])) == 1, name=f'one_option_{fam}')
m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)) <= resource_limits['Weight'], name='total_weight')
m.addConstr(gp.quicksum((budget_use[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_pairs)) <= resource_limits['BudgetUse'], name='total_budget')
m.optimize()