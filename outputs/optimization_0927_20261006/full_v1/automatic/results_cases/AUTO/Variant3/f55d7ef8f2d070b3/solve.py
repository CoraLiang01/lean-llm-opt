import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
for col in ['NormalDuration', 'CrashDuration', 'CrashCostPerDay']:
    activities_df[col] = activities_df[col].astype(int)
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
params_df['Value'] = params_df['Value'].astype(int)
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
activity_ids = activities_df['Activity'].tolist()

def parse_predecessors(cell):
    cell = cell.strip()
    if cell == '':
        return []
    return [pred.strip() for pred in cell.split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_predecessors(row['Predecessors']) for (_, row) in activities_df.iterrows()}
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost_per_day = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
for (act, preds) in predecessors.items():
    for pred in preds:
        if pred not in activity_ids:
            raise ValueError(f"Activity '{act}' has unknown predecessor '{pred}'")
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, lb=0, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    max_crash = normal_duration[i] - crash_duration[i]
    m.addConstr(z_vars[i] >= 0, name=f'z_lb_{i}')
    m.addConstr(z_vars[i] <= max_crash, name=f'z_ub_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'T_ge_finish_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()