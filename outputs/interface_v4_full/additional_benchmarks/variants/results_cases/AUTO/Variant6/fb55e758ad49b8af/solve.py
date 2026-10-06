import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options_per_family = {f: sorted(option_df[option_df['Family'] == f]['Option'].unique()) for f in families}
value = {}
weight = {}
budget = {}
for _, row in option_df.iterrows():
    f = row['Family']
    o = row['Option']
    value[f, o] = int(row['Value'])
    weight[f, o] = int(row['Weight'])
    budget[f, o] = int(row['BudgetUse'])

def get_limit(resource_name):
    match = limits_df[limits_df['Resource'].str.casefold() == resource_name.casefold()]
    if match.empty:
        raise ValueError(f"Resource limit for '{resource_name}' not found in resource_limits.csv")
    return int(match['Limit'].iloc[0])
weight_limit = get_limit('Weight')
budget_limit = get_limit('BudgetUse')

def solve_bundle_knapsack(families, options_per_family, value, weight, budget, weight_limit, budget_limit):
    m = gp.Model('multi_choice_knapsack')
    x = {}
    for f in families:
        for o in options_per_family[f]:
            x[f, o] = m.addVar(vtype=gp.GRB.BINARY, name=f'x_{f}_{o}')
    m.setObjective(gp.quicksum((value[f, o] * x[f, o] for f in families for o in options_per_family[f])), gp.GRB.MAXIMIZE)
    for f in families:
        m.addConstr(gp.quicksum((x[f, o] for o in options_per_family[f])) == 1, name=f'one_option_{f}')
    m.addConstr(gp.quicksum((weight[f, o] * x[f, o] for f in families for o in options_per_family[f])) <= weight_limit, name='weight_limit')
    m.addConstr(gp.quicksum((budget[f, o] * x[f, o] for f in families for o in options_per_family[f])) <= budget_limit, name='budget_limit')
    m.optimize()
    return m
m = solve_bundle_knapsack(families, options_per_family, value, weight, budget, weight_limit, budget_limit)