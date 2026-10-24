import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_path = 'benchmark_dataset/Variant3/inputs/project_activities.csv'
activities_df = pd.read_csv(activities_path, sep=',', dtype=str, keep_default_na=False)
parameters_path = 'benchmark_dataset/Variant3/inputs/project_parameters.csv'
parameters_df = pd.read_csv(parameters_path, sep=',', dtype=str, keep_default_na=False)
activities_df['Activity'] = activities_df['Activity'].str.strip()
activity_ids = activities_df['Activity'].tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))

def parse_predecessors(s):
    s = s.strip()
    if s == '':
        return []
    return [x.strip() for x in s.split(';') if x.strip() != '']
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds = parse_predecessors(row['Predecessors'])
    predecessors[act] = preds
parameters_df['Parameter'] = parameters_df['Parameter'].str.strip().str.casefold()
deadline_row = parameters_df[parameters_df['Parameter'] == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = {}
for i in activity_ids:
    lb = 0
    ub = normal_duration[i] - crash_duration[i]
    z_vars[i] = m.addVar(lb=lb, ub=ub, vtype=gp.GRB.INTEGER, name=f'z_{i}')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(s_vars[i] + (normal_duration[i] - z_vars[i]) <= T_var, name=f'finish_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()