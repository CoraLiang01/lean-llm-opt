import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
families = option_df['Family'].unique().tolist()
options = option_df['Option'].unique().tolist()
family_options = option_df.groupby('Family')['Option'].apply(list).to_dict()
value_dict = {}
weight_dict = {}
budgetuse_dict = {}
for (idx, row) in option_df.iterrows():
    f = row['Family']
    o = row['Option']
    value_dict[f, o] = row['Value']
    weight_dict[f, o] = row['Weight']
    budgetuse_dict[f, o] = row['BudgetUse']
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Resource_norm'] = limits_df['Resource'].str.strip().str.casefold()
limits_df['Limit'] = limits_df['Limit'].astype(int)

def get_limit(resource_name):
    norm = resource_name.strip().casefold()
    matches = limits_df[limits_df['Resource_norm'] == norm]
    if len(matches) != 1:
        raise ValueError(f"Resource limit for '{resource_name}' not found or not unique.")
    return int(matches.iloc[0]['Limit'])
shelf_weight_limit = get_limit('Weight')
budget_use_limit = get_limit('BudgetUse')
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(f, o) for f in families for o in family_options[f]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[f, o] * x_vars[f, o] for f in families for o in family_options[f])), gp.GRB.MAXIMIZE)
for f in families:
    m.addConstr(gp.quicksum((x_vars[f, o] for o in family_options[f])) == 1, name=f'select_one_{f}')
m.addConstr(gp.quicksum((weight_dict[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= shelf_weight_limit, name='shelf_weight_limit')
m.addConstr(gp.quicksum((budgetuse_dict[f, o] * x_vars[f, o] for f in families for o in family_options[f])) <= budget_use_limit, name='budget_use_limit')
m.optimize()