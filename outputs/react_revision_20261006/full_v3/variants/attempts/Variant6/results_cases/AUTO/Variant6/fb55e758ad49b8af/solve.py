import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
df_options = pd.read_csv(option_catalog_path, sep=',')
df_options['Family_id'] = df_options['Family'].astype(str).str.strip()
df_options['Option_id'] = df_options['Option'].astype(str).str.strip()
df_limits = pd.read_csv(resource_limits_path, sep=',')
df_limits['Resource_id'] = df_limits['Resource'].astype(str).str.strip()
families = sorted(df_options['Family_id'].unique())
options_per_family = {f: sorted(df_options[df_options['Family_id'] == f]['Option_id'].unique()) for f in families}
famopt_keys = [(f, o) for f in families for o in options_per_family[f]]
value = {}
weight = {}
budgetuse = {}
for (_, row) in df_options.iterrows():
    f = row['Family_id']
    o = row['Option_id']
    key = (f, o)
    value[key] = int(row['Value'])
    weight[key] = int(row['Weight'])
    budgetuse[key] = int(row['BudgetUse'])

def get_limit(resource_name):
    matches = df_limits[df_limits['Resource_id'].str.casefold() == resource_name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Resource limit for '{resource_name}' not found or not unique.")
    return int(matches.iloc[0]['Limit'])
weight_limit = get_limit('Weight')
budget_limit = get_limit('BudgetUse')
for key in famopt_keys:
    if key not in value or key not in weight or key not in budgetuse:
        raise ValueError(f'Missing coefficients for {key}')

def solve_bundle_knapsack(famopt_keys, families, options_per_family, value, weight, budgetuse, weight_limit, budget_limit):
    m = gp.Model('bundle_knapsack')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(famopt_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[key] * x[key] for key in famopt_keys)), gp.GRB.MAXIMIZE)
    for f in families:
        m.addConstr(gp.quicksum((x[f, o] for o in options_per_family[f])) == 1, name='family_sel')
    m.addConstr(gp.quicksum((weight[key] * x[key] for key in famopt_keys)) <= weight_limit, name='weight_limit')
    m.addConstr(gp.quicksum((budgetuse[key] * x[key] for key in famopt_keys)) <= budget_limit, name='budget_limit')
    m.optimize()
    return m
m = solve_bundle_knapsack(famopt_keys=famopt_keys, families=families, options_per_family=options_per_family, value=value, weight=weight, budgetuse=budgetuse, weight_limit=weight_limit, budget_limit=budget_limit)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')