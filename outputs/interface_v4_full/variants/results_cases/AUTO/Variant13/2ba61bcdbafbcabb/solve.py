import gurobipy as gp
import pandas as pd
import numpy as np
import re
activities_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant13/inputs/project_activities.csv', sep=',')
params_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/benchmark_dataset 3/Variant13/inputs/project_parameters.csv', sep=',')
activities_df['Activity'] = activities_df['Activity'].astype(str).str.strip()
activities_df['Predecessors'] = activities_df['Predecessors'].astype(str).str.strip()
activities = list(activities_df['Activity'])
normal_duration = dict(zip(activities_df['Activity'], activities_df['NormalDuration']))
crash_duration = dict(zip(activities_df['Activity'], activities_df['CrashDuration']))
crash_cost = dict(zip(activities_df['Activity'], activities_df['CrashCostPerDay']))
max_crash_days = {i: normal_duration[i] - crash_duration[i] for i in activities}

def parse_predecessors(cell):
    cell = str(cell).strip()
    if cell == '' or cell.lower() == 'nan':
        return []
    return [pred.strip() for pred in cell.split(';') if pred.strip() != '']
predecessors = {row['Activity']: parse_predecessors(row['Predecessors']) for _, row in activities_df.iterrows()}
successors = {i: [] for i in activities}
for i in activities:
    for pred in predecessors[i]:
        if pred not in successors:
            raise ValueError(f"Predecessor '{pred}' listed for activity '{i}' not found in activity list.")
        successors[pred].append(i)
terminal_activities = [i for i in activities if len(successors[i]) == 0]
deadline_row = params_df[params_df['Parameter'].str.strip().str.casefold() == 'projectdeadline']
if deadline_row.empty:
    raise ValueError('ProjectDeadline parameter not found in project_parameters.csv')
project_deadline = int(deadline_row['Value'].iloc[0])
m = gp.Model('ProjectCrashing')
s = m.addVars(activities, lb=0.0, vtype=gp.GRB.CONTINUOUS, name='')
z = m.addVars(activities, lb=0, ub=[max_crash_days[i] for i in activities], vtype=gp.GRB.INTEGER, name='')
T = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='T')
m.setObjective(gp.quicksum((crash_cost[i] * z[i] for i in activities)), gp.GRB.MINIMIZE)
for i in activities:
    for j in predecessors[i]:
        m.addConstr(s[i] >= s[j] + (normal_duration[j] - z[j]), name=f'prec_{j}_to_{i}')
for i in terminal_activities:
    m.addConstr(T >= s[i] + (normal_duration[i] - z[i]), name=f'completion_{i}')
m.addConstr(T <= project_deadline, name='deadline')
m.optimize()