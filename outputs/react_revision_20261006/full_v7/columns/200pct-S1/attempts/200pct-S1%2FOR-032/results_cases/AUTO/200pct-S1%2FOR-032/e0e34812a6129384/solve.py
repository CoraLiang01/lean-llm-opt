import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
or_mask = courses_df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
or_course_ids = list(or_courses_df['course_id'])
try:
    credits = {cid: int(or_courses_df.loc[or_courses_df['course_id'] == cid, 'credits'].values[0]) for cid in or_course_ids}
    interest_points = {cid: int(or_courses_df.loc[or_courses_df['course_id'] == cid, 'interest_points'].values[0]) for cid in or_course_ids}
except Exception as e:
    raise ValueError(f'Error extracting numeric parameters for Operations Research courses: {e}')
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in parameter extraction for Operations Research courses.')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for cid in or_course_ids:
        print(f'{x_vars[cid].VarName} {x_vars[cid].X}')
else:
    print(f'Solver status: {m.Status}')