import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {g: sorted(option_df[option_df['Family'] == g]['Option'].unique()) for g in families}
value_dict = {}
weight_dict = {}
budgetuse_dict = {}
for idx, row in option_df.iterrows():
    g = row['Family']
    o = row['Option']
    value_dict[g, o] = row['Value']
    weight_dict[g, o] = row['Weight']
    budgetuse_dict[g, o] = row['BudgetUse']
limits_map = {row['Resource'].strip().casefold(): row['Limit'] for _, row in limits_df.iterrows()}
if 'weight' not in limits_map or 'budgetuse' not in limits_map:
    raise ValueError("Resource limits for 'Weight' and 'BudgetUse' must be present in resource_limits.csv")
weight_limit = limits_map['weight']
budgetuse_limit = limits_map['budgetuse']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(g, o) for g in families for o in family_options[g]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x_vars[g, o] for o in family_options[g])) == 1)
m.addConstr(gp.quicksum((weight_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])) <= weight_limit)
m.addConstr(gp.quicksum((budgetuse_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])) <= budgetuse_limit)
m.optimize()