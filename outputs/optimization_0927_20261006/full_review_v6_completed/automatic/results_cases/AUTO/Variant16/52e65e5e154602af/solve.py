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
for (idx, row) in option_df.iterrows():
    g = row['Family']
    o = row['Option']
    value_dict[g, o] = row['Value']
    weight_dict[g, o] = row['Weight']
    budgetuse_dict[g, o] = row['BudgetUse']
resource_limits = {}
for (idx, row) in limits_df.iterrows():
    resource = row['Resource']
    limit = row['Limit']
    resource_limits[resource] = limit
for res in ['Weight', 'BudgetUse']:
    if res not in resource_limits:
        raise ValueError(f"Resource limit for '{res}' not found in resource_limits.csv.")
m = gp.Model('MultiChoiceKnapsack')
x_vars = m.addVars([(g, o) for g in families for o in family_options[g]], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])), gp.GRB.MAXIMIZE)
for g in families:
    m.addConstr(gp.quicksum((x_vars[g, o] for o in family_options[g])) == 1, name=f'one_option_{g}')
m.addConstr(gp.quicksum((weight_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])) <= resource_limits['Weight'], name='total_weight')
m.addConstr(gp.quicksum((budgetuse_dict[g, o] * x_vars[g, o] for g in families for o in family_options[g])) <= resource_limits['BudgetUse'], name='total_budgetuse')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value: {m.objVal:.0f}')
    print('--- Selected Options ---')
    for g in families:
        for o in family_options[g]:
            if x_vars[g, o].X > 0.5:
                print(f'Family {g}: Option {o} (Value={value_dict[g, o]}, Weight={weight_dict[g, o]}, BudgetUse={budgetuse_dict[g, o]})')
    total_weight = sum((weight_dict[g, o] * x_vars[g, o].X for g in families for o in family_options[g]))
    total_budget = sum((budgetuse_dict[g, o] * x_vars[g, o].X for g in families for o in family_options[g]))
    print(f"Total Weight Used: {total_weight:.0f} / {resource_limits['Weight']}")
    print(f"Total Budget Used: {total_budget:.0f} / {resource_limits['BudgetUse']}")
else:
    print(f'No optimal solution found. Status: {m.status}')