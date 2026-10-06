import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
discipline_col = 'discipline'
or_mask = df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
or_courses_df = df[or_mask].copy()
OR_COURSES = list(or_courses_df['course_id'])
if len(OR_COURSES) == 0:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
credits = or_courses_df.set_index('course_id')['credits'].to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].to_dict()
if set(OR_COURSES) != set(credits.keys()) or set(OR_COURSES) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = gp.Model('OR_Course_Selection')
x = m.addVars(OR_COURSES, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in OR_COURSES)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in OR_COURSES)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in OR_COURSES if x[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    total_interest = sum((interest_points[i] for i in selected))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total interest points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for i in selected:
        cname = or_courses_df.loc[or_courses_df['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} | Credits: {credits[i]}, Interest Points: {interest_points[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')