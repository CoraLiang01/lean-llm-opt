import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/option_catalog.csv'
option_df = pd.read_csv(option_catalog_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Value', 'Weight', 'BudgetUse']:
    option_df[col] = option_df[col].astype(int)
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant6/inputs/resource_limits.csv'
limits_df = pd.read_csv(resource_limits_path, sep=',', dtype=str, keep_default_na=False)
limits_df['Limit'] = limits_df['Limit'].astype(int)
families = sorted(option_df['Family'].unique())
family_options = {fam: sorted(option_df[option_df['Family'] == fam]['Option'].unique()) for fam in families}
value_dict = {}
weight_dict = {}
budgetuse_dict = {}
for (idx, row) in option_df.iterrows():
    fam = row['Family']
    opt = row['Option']
    value_dict[fam, opt] = row['Value']
    weight_dict[fam, opt] = row['Weight']
    budgetuse_dict[fam, opt] = row['BudgetUse']
limits_df['Resource_norm'] = limits_df['Resource'].str.strip().str.casefold()
resource_limit = {}
for (idx, row) in limits_df.iterrows():
    resource_limit[row['Resource_norm']] = row['Limit']
weight_limit = resource_limit.get('weight')
budget_limit = resource_limit.get('budgetuse')
if weight_limit is None or budget_limit is None:
    raise ValueError("Missing required resource limits for 'Weight' or 'BudgetUse'.")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(fam, opt) for fam in families for opt in family_options[fam]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])), gp.GRB.MAXIMIZE)
for fam in families:
    m.addConstr(gp.quicksum((x_vars[fam, opt] for opt in family_options[fam])) == 1, name=f'select_one_{fam}')
m.addConstr(gp.quicksum((weight_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budgetuse_dict[fam, opt] * x_vars[fam, opt] for fam in families for opt in family_options[fam])) <= budget_limit, name='budget_limit')
m.optimize()