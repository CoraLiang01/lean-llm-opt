import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str)
option_df['Option'] = option_df['Option'].astype(str)
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str)
families = sorted(option_df['Family'].unique())
options_by_family = {g: sorted(option_df[option_df['Family'] == g]['Option'].unique()) for g in families}
family_option_pairs = [(row['Family'], row['Option']) for _, row in option_df.iterrows()]
value = {(row['Family'], row['Option']): int(row['Value']) for _, row in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for _, row in option_df.iterrows()}
budget_use = {(row['Family'], row['Option']): int(row['BudgetUse']) for _, row in option_df.iterrows()}
resource_limits = {row['Resource']: int(row['Limit']) for _, row in limits_df.iterrows()}
if 'Weight' not in resource_limits or 'BudgetUse' not in resource_limits:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
weight_limit = resource_limits['Weight']
budget_limit = resource_limits['BudgetUse']
m = gp.Model('MultiChoiceKnapsack')
x = m.addVars(family_option_pairs, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[g, o] * x[g, o] for g, o in family_option_pairs)), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x[g, o] for o in options_by_family[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x[g, o] for g, o in family_option_pairs)) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budget_use[g, o] * x[g, o] for g, o in family_option_pairs)) <= budget_limit, name='budget_limit')
m.optimize()