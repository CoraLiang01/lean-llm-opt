import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
or_mask = df['discipline'].astype(str).str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = or_courses_df['course_id'].astype(str).tolist()
credits = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['credits'].astype(int)))
interest_points = dict(zip(or_courses_df['course_id'].astype(str), or_courses_df['interest_points'].astype(int)))
if set(or_course_ids) != set(credits.keys()) or set(or_course_ids) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in or_course_ids if x[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits} / 20')
    print('\nSelected Operations Research courses:')
    for i in selected:
        course_row = or_courses_df[or_courses_df['course_id'] == i].iloc[0]
        print(f"  {i}: {course_row['course_name']} (Credits: {credits[i]}, Interest Points: {interest_points[i]})")
else:
    print(f'No optimal solution found. Status: {m.status}')