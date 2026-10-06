import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
family_options = {g: sorted(option_df[option_df['Family'] == g]['Option'].unique()) for g in families}
value = {}
weight = {}
budgetuse = {}
for _, row in option_df.iterrows():
    g = str(row['Family']).strip()
    o = str(row['Option']).strip()
    value[g, o] = int(row['Value'])
    weight[g, o] = int(row['Weight'])
    budgetuse[g, o] = int(row['BudgetUse'])
resource_limits = {}
for _, row in limits_df.iterrows():
    r = str(row['Resource']).strip()
    resource_limits[r] = int(row['Limit'])
for req_res in ['Weight', 'BudgetUse']:
    if req_res not in resource_limits:
        raise ValueError(f"Resource limit for '{req_res}' not found in resource_limits.csv.")
m = gp.Model('multi_choice_knapsack')
x = {}
for g in families:
    for o in family_options[g]:
        x[g, o] = m.addVar(vtype=gp.GRB.BINARY, name=f'x_{g}_{o}')
m.setObjective(gp.quicksum((value[g, o] * x[g, o] for g in families for o in family_options[g])), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x[g, o] for o in family_options[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight[g, o] * x[g, o] for g in families for o in family_options[g])) <= resource_limits['Weight'], name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[g, o] * x[g, o] for g in families for o in family_options[g])) <= resource_limits['BudgetUse'], name='budgetuse_limit')
m.optimize()