import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant22/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
fam_opt_pairs = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value = {(row['Family'], row['Option']): int(row['Value']) for (_, row) in option_df.iterrows()}
weight = {(row['Family'], row['Option']): int(row['Weight']) for (_, row) in option_df.iterrows()}
labor = {(row['Family'], row['Option']): int(row['LaborHours']) for (_, row) in option_df.iterrows()}

def get_limit(resource_name):
    matches = limits_df[limits_df['Resource'].str.casefold() == resource_name.casefold()]
    if len(matches) != 1:
        raise ValueError(f"Resource limit for '{resource_name}' not found or not unique.")
    return int(matches.iloc[0]['Limit'])
weight_limit = get_limit('Weight')
labor_limit = get_limit('LaborHours')
for key in fam_opt_pairs:
    if key not in value or key not in weight or key not in labor:
        raise ValueError(f'Missing coefficients for family-option pair {key}.')

def solve_problem():
    m = gp.Model('MultiChoiceKnapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[key] * x[key] for key in fam_opt_pairs)), gp.GRB.MAXIMIZE)
    for c in families:
        m.addConstr(gp.quicksum((x[c, o] for o in option_df.loc[option_df['Family'] == c, 'Option'])) == 1, name='oneopt_' + c)
    m.addConstr(gp.quicksum((weight[key] * x[key] for key in fam_opt_pairs)) <= weight_limit, name='weight')
    m.addConstr(gp.quicksum((labor[key] * x[key] for key in fam_opt_pairs)) <= labor_limit, name='labor')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')