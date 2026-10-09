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
options_per_family = {f: sorted(df_options.loc[df_options['Family_id'] == f, 'Option_id'].unique()) for f in families}
famopt_keys = [(f, o) for f in families for o in options_per_family[f]]
value = {}
weight = {}
budgetuse = {}
for (_, row) in df_options.iterrows():
    f = row['Family_id']
    o = row['Option_id']
    value[f, o] = int(row['Value'])
    weight[f, o] = int(row['Weight'])
    budgetuse[f, o] = int(row['BudgetUse'])
for k in famopt_keys:
    if k not in value or k not in weight or k not in budgetuse:
        raise ValueError(f'Missing parameter for family-option pair {k}')
resource_limit = {r.strip().casefold(): int(lim) for (r, lim) in zip(df_limits['Resource_id'], df_limits['Limit'])}
if 'weight' not in resource_limit or 'budgetuse' not in resource_limit:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
weight_limit = resource_limit['weight']
budget_limit = resource_limit['budgetuse']
m = gp.Model('multi_choice_knapsack')
x = m.addVars(famopt_keys, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[k] * x[k] for k in famopt_keys)), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x[f, o] for o in options_per_family[f])) == 1, name=f'family_{f}')
m.addConstr(gp.quicksum((weight[k] * x[k] for k in famopt_keys)) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[k] * x[k] for k in famopt_keys)) <= budget_limit, name='budget_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for k in famopt_keys:
        print(f'{x[k].VarName} {x[k].X}')
else:
    print(f'Solver status: {m.status}')