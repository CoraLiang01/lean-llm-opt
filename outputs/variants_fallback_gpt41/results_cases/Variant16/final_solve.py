import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = 'benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = 'benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {g: sorted(option_df[option_df['Family'] == g]['Option'].unique()) for g in families}
go_pairs = [(g, o) for g in families for o in family_options[g]]
value = {(row['Family'], row['Option']): row['Value'] for (_, row) in option_df.iterrows()}
weight = {(row['Family'], row['Option']): row['Weight'] for (_, row) in option_df.iterrows()}
budget_use = {(row['Family'], row['Option']): row['BudgetUse'] for (_, row) in option_df.iterrows()}

def norm(s):
    return s.strip().casefold()
resource_limits = {}
for (_, row) in limits_df.iterrows():
    resource_limits[norm(row['Resource'])] = row['Limit']
if 'weight' not in resource_limits or 'budgetuse' not in resource_limits:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
weight_limit = resource_limits['weight']
budget_limit = resource_limits['budgetuse']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(go_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x_vars[g, o] for (g, o) in go_pairs)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x_vars[g, o] for o in family_options[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x_vars[g, o] for (g, o) in go_pairs)) <= weight_limit, name='total_weight')
m.addConstr(gp.quicksum((budget_use[g, o] * x_vars[g, o] for (g, o) in go_pairs)) <= budget_limit, name='total_budget')
m.optimize()