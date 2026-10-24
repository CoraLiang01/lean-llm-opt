import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = 'benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
resource_limits_path = 'benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
value = {}
weight = {}
budgetuse = {}
for (_, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value[fam, opt] = row['Value']
    weight[fam, opt] = row['Weight']
    budgetuse[fam, opt] = row['BudgetUse']
limits_dict = {}
for (_, row) in limits_df.iterrows():
    resource = row['Resource'].strip().casefold()
    limits_dict[resource] = row['Limit']
if 'weight' not in limits_dict or 'budgetuse' not in limits_dict:
    raise ValueError("Missing required resource limits for 'Weight' and/or 'BudgetUse'.")
weight_limit = limits_dict['weight']
budget_limit = limits_dict['budgetuse']
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars(((fam, opt) for fam in families for opt in family_options[fam]), vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'select_one_{fam}')
m.addConstr(gp.quicksum((weight[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budgetuse[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= budget_limit, name='budget_limit')
m.optimize()