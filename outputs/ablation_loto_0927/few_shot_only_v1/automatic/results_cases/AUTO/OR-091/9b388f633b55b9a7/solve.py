import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',')
or_courses_df = courses_df[courses_df['discipline'].astype(str) == 'Operations Research'].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = or_courses_df['course_id'].astype(str).tolist()
credits = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['credits']))
interest_points = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['interest_points']))
for cid in or_course_ids:
    if cid not in credits or cid not in interest_points:
        raise ValueError(f'Missing data for Operations Research course_id: {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[cid] * x[cid] for cid in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[cid] * x[cid] for cid in or_course_ids)) <= 20, name='CreditLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [cid for cid in or_course_ids if x[cid].X > 0.5]
    total_credits = sum((credits[cid] for cid in selected))
    total_interest = sum((interest_points[cid] for cid in selected))
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for cid in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'].astype(str) == cid, 'course_name'].values[0]
        print(f'  {cid}: {cname} | Credits: {credits[cid]} | Interest Points: {interest_points[cid]}')
else:
    print(f'No optimal solution found. Status: {m.status}')