import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip().casefold()
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'LaborHours']:
    option_df[col] = option_df[col].astype(int)
option_df['Family_norm'] = option_df['Family'].apply(norm_str)
option_df['Option_norm'] = option_df['Option'].apply(norm_str)
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Resource_norm'] = limits_df['Resource'].apply(norm_str)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options_per_family = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
family_option_tuples = list(option_df[['Family', 'Option']].itertuples(index=False, name=None))
value_dict = {(row['Family'], row['Option']): row['Value'] for (_, row) in option_df.iterrows()}
weight_dict = {(row['Family'], row['Option']): row['Weight'] for (_, row) in option_df.iterrows()}
labor_dict = {(row['Family'], row['Option']): row['LaborHours'] for (_, row) in option_df.iterrows()}
resource_limit_dict = {row['Resource_norm']: row['Limit'] for (_, row) in limits_df.iterrows()}
weight_limit = None
labor_limit = None
for (res_norm, lim) in resource_limit_dict.items():
    if res_norm == norm_str('Weight'):
        weight_limit = lim
    elif res_norm == norm_str('LaborHours'):
        labor_limit = lim
if weight_limit is None or labor_limit is None:
    raise ValueError("Missing required resource limits for 'Weight' or 'LaborHours'.")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(family_option_tuples, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[fam, opt] * x_vars[fam, opt] for (fam, opt) in family_option_tuples)), gp.GRB.MAXIMIZE)
for fam in families:
    fam_options = options_per_family[fam]
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in fam_options)) == 1, name=f'one_option_{fam}')
m.addConstr(gp.quicksum((weight_dict[fam, opt] * x_vars[fam, opt] for (fam, opt) in family_option_tuples)) <= weight_limit, name='total_weight')
m.addConstr(gp.quicksum((labor_dict[fam, opt] * x_vars[fam, opt] for (fam, opt) in family_option_tuples)) <= labor_limit, name='total_labor')
m.optimize()