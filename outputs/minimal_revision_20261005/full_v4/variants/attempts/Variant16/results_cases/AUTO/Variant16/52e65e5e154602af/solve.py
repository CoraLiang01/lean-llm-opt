import gurobipy as gp
import pandas as pd
import numpy as np
option_catalog_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/option_catalog.csv'
resource_limits_path = '/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant16/inputs/resource_limits.csv'
option_df = pd.read_csv(option_catalog_path, sep=',')
option_df['Family'] = option_df['Family'].astype(str).str.strip()
option_df['Option'] = option_df['Option'].astype(str).str.strip()
limits_df = pd.read_csv(resource_limits_path, sep=',')
limits_df['Resource'] = limits_df['Resource'].astype(str).str.strip()
families = sorted(option_df['Family'].unique())
options = sorted(option_df['Option'].unique())
famopt_tuples = [(row['Family'], row['Option']) for (_, row) in option_df.iterrows()]
value_dict = {(row['Family'], row['Option']): int(row['Value']) for (_, row) in option_df.iterrows()}
weight_dict = {(row['Family'], row['Option']): int(row['Weight']) for (_, row) in option_df.iterrows()}
budget_dict = {(row['Family'], row['Option']): int(row['BudgetUse']) for (_, row) in option_df.iterrows()}
weight_limit_row = limits_df.loc[limits_df['Resource'].str.casefold() == 'weight']
budget_limit_row = limits_df.loc[limits_df['Resource'].str.casefold() == 'budgetuse']
if weight_limit_row.empty or budget_limit_row.empty:
    raise ValueError("Missing resource limits for 'Weight' or 'BudgetUse' in resource_limits.csv.")
weight_limit = int(weight_limit_row['Limit'].iloc[0])
budget_limit = int(budget_limit_row['Limit'].iloc[0])
for tup in famopt_tuples:
    if tup not in value_dict or tup not in weight_dict or tup not in budget_dict:
        raise ValueError(f'Missing coefficients for family-option pair {tup}.')
m = gp.Model('multi_choice_knapsack')
x = m.addVars(famopt_tuples, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[g_o] * x[g_o] for g_o in famopt_tuples)), gp.GRB.MAXIMIZE)
for g in families:
    fam_options = [(g, o) for o in option_df.loc[option_df['Family'] == g, 'Option']]
    m.addConstr(gp.quicksum((x[g_o] for g_o in fam_options)) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight_dict[g_o] * x[g_o] for g_o in famopt_tuples)) <= weight_limit, name='weight_limit')
m.addConstr(gp.quicksum((budget_dict[g_o] * x[g_o] for g_o in famopt_tuples)) <= budget_limit, name='budget_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for g_o in famopt_tuples:
        var = x[g_o]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')