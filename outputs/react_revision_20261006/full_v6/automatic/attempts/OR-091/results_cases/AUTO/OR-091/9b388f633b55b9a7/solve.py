import gurobipy as gp
import pandas as pd
import numpy as np
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
or_mask = courses_df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    try:
        credits[cid] = int(row['credits'])
    except Exception:
        raise ValueError(f'Invalid or missing credits for course_id {cid}')
    try:
        interest_points[cid] = int(row['interest_points'])
    except Exception:
        raise ValueError(f'Invalid or missing interest_points for course_id {cid}')
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')

def solve_or_knapsack(or_course_ids, credits, interest_points):
    m = gp.Model('OR_Course_Selection')
    x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_or_knapsack(or_course_ids, credits, interest_points)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')