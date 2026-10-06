import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset/Variant3/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = activities_df['Activity'].tolist()
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
predecessors = {}
for (idx, row) in activities_df.iterrows():
    act = row['Activity']
    preds_raw = str(row['Predecessors']).strip()
    if preds_raw == '' or preds_raw.lower() == 'nan':
        predecessors[act] = []
    else:
        preds = [p.strip() for p in preds_raw.split(';') if p.strip() != '']
        predecessors[act] = preds
successors = {a: [] for a in activities}
for i in activities:
    for j in predecessors[i]:
        if j not in successors:
            raise ValueError(f"Predecessor '{j}' for activity '{i}' not found in activity list.")
        successors[j].append(i)
terminal_activities = [a for a in activities if len(successors[a]) == 0]
deadline_row = params_df[params_df['Parameter'].str.casefold().str.strip() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
for a in activities:
    if a not in normal_duration or a not in crash_duration or a not in crash_cost:
        raise ValueError(f"Missing duration or cost data for activity '{a}'.")

def solve_project_crashing():
    m = gp.Model('ProjectCrashing')
    m.setParam('MIPGap', 0.0001)
    s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
    z = m.addVars(activities, lb=0, ub={a: normal_duration[a] - crash_duration[a] for a in activities}, vtype=gp.GRB.INTEGER, name='')
    T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
    m.setObjective(gp.quicksum((crash_cost[a] * z[a] for a in activities)), gp.GRB.MINIMIZE)
    for i in activities:
        for j in predecessors[i]:
            m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
    for a in activities:
        m.addConstr(z[a] >= 0, name=f'z_lb_{a}')
        m.addConstr(z[a] <= normal_duration[a] - crash_duration[a], name=f'z_ub_{a}')
    for a in terminal_activities:
        m.addConstr(T >= s[a] + (normal_duration[a] - z[a]), name=f'T_terminal_{a}')
    m.addConstr(T <= project_deadline, name='T_deadline')
    for a in activities:
        m.addConstr(s[a] >= 0, name=f's_lb_{a}')
    m.optimize()
    return m
m = solve_project_crashing()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')