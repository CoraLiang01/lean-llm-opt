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
family_options = {g: sorted(option_df.loc[option_df['Family'] == g, 'Option'].unique()) for g in families}
family_option_pairs = [(g, o) for g in families for o in family_options[g]]
value = {}
weight = {}
budget_use = {}
for (idx, row) in option_df.iterrows():
    g = row['Family']
    o = row['Option']
    value[g, o] = int(row['Value'])
    weight[g, o] = int(row['Weight'])
    budget_use[g, o] = int(row['BudgetUse'])

def get_limit(resource_name):
    mask = limits_df['Resource'].str.casefold().str.strip() == resource_name.casefold().strip()
    matches = limits_df.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"Resource limit for '{resource_name}' not found or not unique in resource_limits.csv")
    return int(matches.iloc[0]['Limit'])
weight_limit = get_limit('Weight')
budget_limit = get_limit('BudgetUse')
for (g, o) in family_option_pairs:
    if (g, o) not in value or (g, o) not in weight or (g, o) not in budget_use:
        raise ValueError(f'Missing coefficients for family {g}, option {o}')

def solve_multichoice_knapsack(families, family_options, family_option_pairs, value, weight, budget_use, weight_limit, budget_limit):
    m = gp.Model('multi_choice_knapsack')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(family_option_pairs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[g, o] * x_vars[g, o] for (g, o) in family_option_pairs)), gp.GRB.MAXIMIZE)
    for g in families:
        m.addConstr(gp.quicksum((x_vars[g, o] for o in family_options[g])) == 1, name=f'one_option_{g}')
    m.addConstr(gp.quicksum((weight[g, o] * x_vars[g, o] for (g, o) in family_option_pairs)) <= weight_limit, name='weight_limit')
    m.addConstr(gp.quicksum((budget_use[g, o] * x_vars[g, o] for (g, o) in family_option_pairs)) <= budget_limit, name='budget_limit')
    m.optimize()
    return m
m = solve_multichoice_knapsack(families, family_options, family_option_pairs, value, weight, budget_use, weight_limit, budget_limit)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for (g, o) in family_option_pairs:
        var = m.getVarByName(f'x[{g},{o}]')
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')