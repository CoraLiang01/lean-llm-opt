import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
or_courses_df = courses_df[courses_df['discipline'] == 'Operations Research'].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'])
credits = {}
interest_points = {}
for (idx, row) in or_courses_df.iterrows():
    cid = row['course_id']
    try:
        credits[cid] = float(row['credits'])
    except Exception:
        raise ValueError(f"Invalid or missing credits for course_id {cid}: {row['credits']}")
    try:
        interest_points[cid] = float(row['interest_points'])
    except Exception:
        raise ValueError(f"Invalid or missing interest_points for course_id {cid}: {row['interest_points']}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[cid] * x_vars[cid].X for cid in or_course_ids))
    total_credits = sum((credits[cid] * x_vars[cid].X for cid in or_course_ids))
    print(f'Optimal total interest points: {total_interest:.2f}')
    print(f'Total credits used: {total_credits:.2f} (limit: 20)')
    print('\nSelected Operations Research courses:')
    for cid in or_course_ids:
        if x_vars[cid].X > 0.5:
            row = or_courses_df[or_courses_df['course_id'] == cid].iloc[0]
            print(f"  {row['course_id']}: {row['course_name']} | Credits: {row['credits']} | Interest Points: {row['interest_points']}")
else:
    print(f'No optimal solution found. Status: {m.status}')