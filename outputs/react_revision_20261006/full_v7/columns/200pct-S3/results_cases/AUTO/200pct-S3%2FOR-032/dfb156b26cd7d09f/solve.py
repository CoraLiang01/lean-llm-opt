import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = df[discipline_col].str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline == 'Operations Research'.")
course_ids = list(or_courses_df['course_id'])
interest_points = {}
credits = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    try:
        ip = int(row['interest_points'])
    except Exception:
        raise ValueError(f'Missing or invalid interest_points for course_id {cid}')
    try:
        cr = int(row['credits'])
    except Exception:
        raise ValueError(f'Missing or invalid credits for course_id {cid}')
    interest_points[cid] = ip
    credits[cid] = cr
if set(interest_points.keys()) != set(course_ids) or set(credits.keys()) != set(course_ids):
    raise ValueError('Mismatch in parameter coverage for Operations Research courses.')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in course_ids)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for cid in course_ids:
        print(f'{x_vars[cid].VarName} {x_vars[cid].X}')
else:
    print(f'Solver status: {m.Status}')