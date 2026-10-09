import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {f: sorted(option_df[option_df['Family'] == f]['Option'].unique()) for f in families}
value = {}
weight = {}
budget_use = {}
for (idx, row) in option_df.iterrows():
    f = row['Family']
    o = row['Option']
    value[f, o] = row['Value']
    weight[f, o] = row['Weight']
    budget_use[f, o] = row['BudgetUse']
limits_dict = {row['Resource'].strip(): row['Limit'] for (idx, row) in limits_df.iterrows()}
if 'Weight' not in limits_dict or 'BudgetUse' not in limits_dict:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
weight_limit = limits_dict['Weight']
budget_limit = limits_dict['BudgetUse']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(f, o) for f in families for o in family_options[f]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[f, o] * x_vars[f, o] for f in families for o in family_options[f])), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x_vars[f, o] for o in family_options[f])) == 1, name=f'SelectOne_{f}')
m.addConstr(gp.quicksum((weight[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= weight_limit, name='WeightLimit')
m.addConstr(gp.quicksum((budget_use[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= budget_limit, name='BudgetLimit')
m.optimize()