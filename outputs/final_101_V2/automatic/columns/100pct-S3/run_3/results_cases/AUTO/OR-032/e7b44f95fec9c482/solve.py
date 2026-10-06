import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
df_or = df[ops_research_mask].copy()
if df_or.empty:
    raise ValueError("No courses found with discipline == 'Operations Research'.")
course_ids = df_or['course_id'].astype(str).tolist()
credits = df_or.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = df_or.set_index('course_id')['interest_points'].astype(int).to_dict()
if set(course_ids) != set(credits.keys()) or set(course_ids) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_ids and parameter keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [i for i in course_ids if x[i].X > 0.5]
    total_interest = sum((interest_points[i] for i in selected_courses))
    total_credits = sum((credits[i] for i in selected_courses))
    print(f'Optimal total value/cost: {m.objVal} (Total interest points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for i in selected_courses:
        course_row = df_or[df_or['course_id'] == i].iloc[0]
        print(f"Course ID: {i}, Name: {course_row['course_name']}, Credits: {credits[i]}, Interest Points: {interest_points[i]}")
else:
    print(f'No optimal solution found. Status: {m.status}')