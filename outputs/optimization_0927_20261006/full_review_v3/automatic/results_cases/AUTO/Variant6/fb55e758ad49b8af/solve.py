import gurobipy as gp
import pandas as pd
import numpy as np
import re
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {fam: sorted(option_df.loc[option_df['Family'] == fam, 'Option'].unique()) for fam in families}
value = {}
weight = {}
budget_use = {}
for (idx, row) in option_df.iterrows():
    f = row['Family']
    o = row['Option']
    value[f, o] = row['Value']
    weight[f, o] = row['Weight']
    budget_use[f, o] = row['BudgetUse']

def norm(s):
    return s.strip().casefold()
limits_dict = {norm(row['Resource']): row['Limit'] for (_, row) in limits_df.iterrows()}
for res in ['Weight', 'BudgetUse']:
    if norm(res) not in limits_dict:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv")
weight_limit = limits_dict[norm('Weight')]
budget_limit = limits_dict[norm('BudgetUse')]
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(f, o) for f in families for o in family_options[f]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[f, o] * x_vars[f, o] for f in families for o in family_options[f])), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x_vars[f, o] for o in family_options[f])) == 1, name='')
m.addConstr(gp.quicksum((weight[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= weight_limit, name='')
m.addConstr(gp.quicksum((budget_use[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= budget_limit, name='')
m.optimize()