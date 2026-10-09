import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
option_df['Family'] = option_df['Family'].str.strip()
option_df['Option'] = option_df['Option'].str.strip()
option_df['Value'] = option_df['Value'].astype(int)
option_df['Weight'] = option_df['Weight'].astype(int)
option_df['BudgetUse'] = option_df['BudgetUse'].astype(int)
limits_df['Resource'] = limits_df['Resource'].str.strip()
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options_per_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
fam_opt_keys = []
for fam in families:
    for opt in options_per_family[fam]:
        fam_opt_keys.append((fam, opt))
value = {}
weight = {}
budget_use = {}
for (fam, opt) in fam_opt_keys:
    row = option_df[(option_df['Family'] == fam) & (option_df['Option'] == opt)]
    if row.shape[0] != 1:
        raise ValueError(f'Missing or duplicate entry for Family={fam}, Option={opt}')
    value[fam, opt] = int(row['Value'].iloc[0])
    weight[fam, opt] = int(row['Weight'].iloc[0])
    budget_use[fam, opt] = int(row['BudgetUse'].iloc[0])

def get_limit(resource_name):
    match = limits_df[limits_df['Resource'].str.casefold() == resource_name.casefold()]
    if match.shape[0] != 1:
        raise ValueError(f"Resource limit for '{resource_name}' not found or not unique.")
    return int(match['Limit'].iloc[0])
weight_limit = get_limit('Weight')
budget_limit = get_limit('BudgetUse')

def solve_bundle_knapsack(fam_opt_keys, families, options_per_family, value, weight, budget_use, weight_limit, budget_limit):
    m = gp.Model('multi_choice_knapsack')
    x_vars = m.addVars(fam_opt_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_keys)), gp.GRB.MAXIMIZE)
    for fam in families:
        m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in options_per_family[fam])) == 1, name=f'select_one_{fam}')
    m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_keys)) <= weight_limit, name='weight_limit')
    m.addConstr(gp.quicksum((budget_use[fam, opt] * x_vars[fam, opt] for (fam, opt) in fam_opt_keys)) <= budget_limit, name='budget_limit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_bundle_knapsack(fam_opt_keys=fam_opt_keys, families=families, options_per_family=options_per_family, value=value, weight=weight, budget_use=budget_use, weight_limit=weight_limit, budget_limit=budget_limit)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')