import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_activities.csv', sep=',', dtype=str, keep_default_na=False)
parameters_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant21/inputs/project_parameters.csv', sep=',', dtype=str, keep_default_na=False)
activity_ids = activities_df['Activity'].astype(str).tolist()
normal_duration = {}
crash_duration = {}
crash_cost_per_day = {}
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = str(row['Activity'])
    try:
        normal_duration[act] = int(row['NormalDuration'])
        crash_duration[act] = int(row['CrashDuration'])
        crash_cost_per_day[act] = int(row['CrashCostPerDay'])
    except Exception as e:
        raise ValueError(f"Non-numeric duration or cost for activity '{act}': {e}")
    preds = row['Predecessors'].strip()
    if preds == '':
        predecessors[act] = []
    else:
        preds_list = [p.strip() for p in preds.split(';') if p.strip() != '']
        predecessors[act] = preds_list
for (act, preds) in predecessors.items():
    for pred in preds:
        if pred not in activity_ids:
            raise ValueError(f"Activity '{act}' references unknown predecessor '{pred}'.")
crash_day_ub = {}
for act in activity_ids:
    ub = normal_duration[act] - crash_duration[act]
    if ub < 0:
        raise ValueError(f"Activity '{act}' has CrashDuration > NormalDuration.")
    crash_day_ub[act] = ub
deadline_row = parameters_df.loc[parameters_df['Parameter'].str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv.')
try:
    project_deadline = int(deadline_row.iloc[0]['Value'])
except Exception as e:
    raise ValueError(f'Non-numeric ProjectDeadline value: {e}')

def solve_project_crashing(activity_ids, normal_duration, crash_duration, crash_cost_per_day, crash_day_ub, predecessors, project_deadline):
    m = gp.Model('ProjectCrashing')
    m.Params.MIPGap = 0.0001
    s_vars = m.addVars(activity_ids, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z_vars = m.addVars(activity_ids, lb=0, ub=[crash_day_ub[i] for i in activity_ids], vtype=gp.GRB.INTEGER, name='')
    T_var = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost_per_day[i] * z_vars[i] for i in activity_ids)), gp.GRB.MINIMIZE)
    for i in activity_ids:
        for j in predecessors[i]:
            m.addConstr(s_vars[i] >= s_vars[j] + (normal_duration[j] - z_vars[j]), name=f'prec_{j}_to_{i}')
    for i in activity_ids:
        m.addConstr(T_var >= s_vars[i] + (normal_duration[i] - z_vars[i]), name=f'compl_{i}')
    m.addConstr(T_var <= project_deadline, name='deadline')
    m.optimize()
    return m
m = solve_project_crashing(activity_ids=activity_ids, normal_duration=normal_duration, crash_duration=crash_duration, crash_cost_per_day=crash_cost_per_day, crash_day_ub=crash_day_ub, predecessors=predecessors, project_deadline=project_deadline)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')