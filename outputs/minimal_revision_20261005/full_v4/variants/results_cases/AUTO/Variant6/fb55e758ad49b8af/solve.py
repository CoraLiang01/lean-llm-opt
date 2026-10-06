import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
df_options = pd.read_csv(option_catalog_path, sep=',')
df_options['Family'] = df_options['Family'].astype(str).str.strip()
df_options['Option'] = df_options['Option'].astype(str).str.strip()
df_limits = pd.read_csv(resource_limits_path, sep=',')
df_limits['Resource'] = df_limits['Resource'].astype(str).str.strip()
families = sorted(df_options['Family'].unique())
options_per_family = {f: sorted(df_options.loc[df_options['Family'] == f, 'Option'].unique()) for f in families}
fam_opt_keys = []
for f in families:
    for o in options_per_family[f]:
        fam_opt_keys.append((f, o))
value = {}
weight = {}
budget_use = {}
for (idx, row) in df_options.iterrows():
    f = row['Family']
    o = row['Option']
    value[f, o] = int(row['Value'])
    weight[f, o] = int(row['Weight'])
    budget_use[f, o] = int(row['BudgetUse'])
for key in fam_opt_keys:
    if key not in value or key not in weight or key not in budget_use:
        raise ValueError(f'Missing data for family-option pair: {key}')
resource_limits = {}
for (idx, row) in df_limits.iterrows():
    res = row['Resource'].strip()
    lim = int(row['Limit'])
    resource_limits[res] = lim
if 'Weight' not in resource_limits or 'BudgetUse' not in resource_limits:
    raise ValueError("Resource limits must include both 'Weight' and 'BudgetUse'.")

def solve_bundle_knapsack(fam_opt_keys, families, options_per_family, value, weight, budget_use, resource_limits):
    m = gp.Model('bundle_knapsack')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(fam_opt_keys, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[f, o] * x[f, o] for (f, o) in fam_opt_keys)), gp.GRB.MAXIMIZE)
    for f in families:
        m.addConstr(gp.quicksum((x[f, o] for o in options_per_family[f])) == 1, name='family_sel_' + f)
    m.addConstr(gp.quicksum((weight[f, o] * x[f, o] for (f, o) in fam_opt_keys)) <= resource_limits['Weight'], name='weight_limit')
    m.addConstr(gp.quicksum((budget_use[f, o] * x[f, o] for (f, o) in fam_opt_keys)) <= resource_limits['BudgetUse'], name='budget_limit')
    m.optimize()
    return m
m = solve_bundle_knapsack(fam_opt_keys=fam_opt_keys, families=families, options_per_family=options_per_family, value=value, weight=weight, budget_use=budget_use, resource_limits=resource_limits)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')