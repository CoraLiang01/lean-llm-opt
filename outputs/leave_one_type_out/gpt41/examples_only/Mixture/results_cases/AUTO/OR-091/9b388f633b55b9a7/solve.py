import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',')
or_courses_df = df[df['discipline'].astype(str).str.casefold().str.strip() == 'operations research']
or_course_ids = list(or_courses_df['course_id'].astype(str))
if len(or_course_ids) == 0:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
credits = or_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
interest_points = or_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
m = gp.Model('OR_Course_Selection')
x = m.addVars(or_course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in or_course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x[i] for i in or_course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[i] for i in or_course_ids if x[i].X > 0.5))
    total_credits = sum((credits[i] for i in or_course_ids if x[i].X > 0.5))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for i in or_course_ids:
        if x[i].X > 0.5:
            course_row = or_courses_df[or_courses_df['course_id'] == i].iloc[0]
            print(f"  {i}: {course_row['course_name']} (Credits: {credits[i]}, Interest Points: {interest_points[i]})")
else:
    print(f'No optimal solution found. Status: {m.status}')