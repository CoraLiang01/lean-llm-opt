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
options_by_family = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
fam_opt_pairs = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value = {}
weight = {}
laborhours = {}
for (_, row) in option_df.iterrows():
    key = (str(row['Family']).strip(), str(row['Option']).strip())
    value[key] = int(row['Value'])
    weight[key] = int(row['Weight'])
    laborhours[key] = int(row['LaborHours'])
resource_limits = {}
for (_, row) in limits_df.iterrows():
    res = str(row['Resource']).strip()
    resource_limits[res] = int(row['Limit'])
for res in ['Weight', 'LaborHours']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")

def solve_multichoice_knapsack(fam_opt_pairs, families, options_by_family, value, weight, laborhours, resource_limits):
    m = gp.Model('multi_choice_knapsack')
    x = m.addVars(fam_opt_pairs, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[c, o] * x[c, o] for (c, o) in fam_opt_pairs)), gp.GRB.MAXIMIZE)
    for c in families:
        m.addConstr(gp.quicksum((x[c, o] for o in options_by_family[c])) == 1, name='oneopt_' + c)
    m.addConstr(gp.quicksum((weight[c, o] * x[c, o] for (c, o) in fam_opt_pairs)) <= resource_limits['Weight'], name='weight_limit')
    m.addConstr(gp.quicksum((laborhours[c, o] * x[c, o] for (c, o) in fam_opt_pairs)) <= resource_limits['LaborHours'], name='laborhours_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_multichoice_knapsack(fam_opt_pairs=fam_opt_pairs, families=families, options_by_family=options_by_family, value=value, weight=weight, laborhours=laborhours, resource_limits=resource_limits)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')