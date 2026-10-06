import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
or_mask = df['discipline'].astype(str).str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'])
credits_dict = dict(zip(or_courses_df['course_id'], or_courses_df['credits']))
interest_points_dict = dict(zip(or_courses_df['course_id'], or_courses_df['interest_points']))
for cid in or_course_ids:
    if cid not in credits_dict or cid not in interest_points_dict:
        raise ValueError(f'Missing credits or interest_points for course_id {cid}')
    if not (isinstance(credits_dict[cid], (int, np.integer)) and isinstance(interest_points_dict[cid], (int, np.integer))):
        raise ValueError(f'Non-integer credits or interest_points for course_id {cid}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points_dict[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in or_course_ids if x[i].X > 0.5]
    total_credits = sum((credits_dict[i] for i in selected))
    total_interest = sum((interest_points_dict[i] for i in selected))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits} / 20')
    print('\nSelected Operations Research courses:')
    for i in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} (Credits: {credits_dict[i]}, Interest Points: {interest_points_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')