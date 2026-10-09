import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant13/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
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
    return [pred.strip() for pred in s.split(';') if pred.strip() != '']
predecessors = dict(zip(activities_df['Activity'], activities_df['Predecessors'].apply(parse_predecessors)))
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
max_crash_days = {i: normal_duration[i] - crash_duration[i] for i in activity_ids}
z_vars = m.addVars(activity_ids, lb=0, ub=[max_crash_days[i] for i in activity_ids], vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'completion_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()