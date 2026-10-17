import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).str.strip()
activity_set = list(activity_ids)

def to_int_series(df, col):
    return df[col].astype(str).str.strip().astype(int)
normal_duration = dict(zip(activity_ids, to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activity_ids, to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activity_ids, to_int_series(activities_df, 'CrashCostPerDay')))

def parse_predecessors(s):
    s = str(s).strip()
    if s == '':
        return []
    return [x.strip() for x in s.split(';') if x.strip() != '']
predecessors = dict(zip(activity_ids, activities_df['Predecessors'].apply(parse_predecessors)))
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if len(deadline_row) != 1:
    raise ValueError('ProjectDeadline parameter not found or not unique in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_set, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_bounds = {i: (0, normal_duration[i] - crash_duration[i]) for i in activity_set}
z_vars = m.addVars(activity_set, lb={i: z_bounds[i][0] for i in activity_set}, ub={i: z_bounds[i][1] for i in activity_set}, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_set)), gp.GRB.MINIMIZE)
for i in activity_set:
    for j in predecessors[i]:
        if j not in activity_set:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_set:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='project_deadline')
m.optimize()