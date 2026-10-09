import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).tolist()

def to_int_series(df, col):
    return df[col].astype(int)
normal_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'NormalDuration')))
crash_duration = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashDuration')))
crash_cost_per_day = dict(zip(activities_df['Activity'], to_int_series(activities_df, 'CrashCostPerDay')))
crash_bounds = {act: (0, normal_duration[act] - crash_duration[act]) for act in activity_ids}

def parse_predecessors(s):
    s = s.strip()
    if s == '':
        return []
    return [x.strip() for x in re.split('[;,\\s]+', s) if x.strip() != '']
predecessors = {row['Activity']: parse_predecessors(row['Predecessors']) for (_, row) in activities_df.iterrows()}
deadline_row = parameters_df[parameters_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row.iloc[0]['Value'])
m = gp.Model('ProjectCrashing')
s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z_vars = m.addVars(activity_ids, vtype=gp.GRB.INTEGER, name='')
T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
for act in activity_ids:
    (lb, ub) = crash_bounds[act]
    z_vars[act].setAttr(gp.GRB.Attr.LB, lb)
    z_vars[act].setAttr(gp.GRB.Attr.UB, ub)
m.setObjective(gp.quicksum((crash_cost_per_day[act] * z_vars[act] for act in activity_ids)), gp.GRB.MINIMIZE)
for i in activity_ids:
    for j in predecessors[i]:
        if j not in activity_ids:
            raise ValueError(f"Predecessor '{j}' of activity '{i}' not found in activity list.")
        m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
for i in activity_ids:
    m.addConstr(s_vars[i] + (normal_duration[i] - z_vars[i]) <= T_var, name=f'finish_{i}')
m.addConstr(T_var <= project_deadline, name='deadline')
m.optimize()