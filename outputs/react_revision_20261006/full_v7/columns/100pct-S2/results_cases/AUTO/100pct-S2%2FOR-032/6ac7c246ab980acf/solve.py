import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found with discipline 'Operations Research'.")
course_id_col = 'course_id'
interest_points_col = 'interest_points'
credits_col = 'credits'
or_course_ids = list(or_courses_df[course_id_col])
interest_points = {}
credits = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row[course_id_col]
    try:
        ip = int(row[interest_points_col])
        cr = int(row[credits_col])
    except Exception as e:
        raise ValueError(f'Non-numeric value in interest_points or credits for course_id {cid}: {e}')
    interest_points[cid] = ip
    credits[cid] = cr
if set(interest_points.keys()) != set(or_course_ids) or set(credits.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in parameter keys for Operations Research courses.')
CREDIT_LIMIT = 20
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= CREDIT_LIMIT, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for cid in or_course_ids:
        print(f'{x_vars[cid].VarName} {x_vars[cid].X}')
else:
    print(f'Solver status: {m.Status}')