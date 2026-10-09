import gurobipy as gp
import pandas as pd
import numpy as np
import re

def normalize_str(s):
    return str(s).strip().casefold()
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
option_df['Family_norm'] = option_df['Family'].apply(normalize_str)
option_df['Option_norm'] = option_df['Option'].apply(normalize_str)
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Resource_norm'] = limits_df['Resource'].apply(normalize_str)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
family_set = sorted(option_df['Family'].unique())
family_to_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in family_set}
value_dict = {}
weight_dict = {}
budgetuse_dict = {}
for (_, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value_dict[fam, opt] = row['Value']
    weight_dict[fam, opt] = row['Weight']
    budgetuse_dict[fam, opt] = row['BudgetUse']
resource_limit_dict = dict(zip(limits_df['Resource_norm'], limits_df['Limit']))
weight_limit = None
budgetuse_limit = None
for (res_norm, lim) in resource_limit_dict.items():
    if res_norm == 'weight':
        weight_limit = lim
    elif res_norm == 'budgetuse':
        budgetuse_limit = lim
if weight_limit is None or budgetuse_limit is None:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(((fam, opt) for fam in family_set for opt in family_to_options[fam]), vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[fam, opt] * x_vars[fam, opt] for fam in family_set for opt in family_to_options[fam])), gp.GRB.MAXIMIZE)
for fam in family_set:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_to_options[fam])) == 1, name=f'select_one_{fam}')
m.addConstr(gp.quicksum((weight_dict[fam, opt] * x_vars[fam, opt] for fam in family_set for opt in family_to_options[fam])) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budgetuse_dict[fam, opt] * x_vars[fam, opt] for fam in family_set for opt in family_to_options[fam])) <= budgetuse_limit, name='budgetuse_limit')
m.optimize()