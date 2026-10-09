import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_courses = df[ops_research_mask].copy()
if ops_research_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_courses['course_id'].tolist()
try:
    credits = ops_research_courses.set_index('course_id')['credits'].astype(int).to_dict()
    interest_points = ops_research_courses.set_index('course_id')['interest_points'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='TotalCredits')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [i for i in course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected_courses))
    total_interest = sum((interest_points[i] for i in selected_courses))
    print(f'Optimal total interest points: {m.objVal:.0f}')
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for i in selected_courses:
        course_row = ops_research_courses[ops_research_courses['course_id'] == i].iloc[0]
        print(f"  {i}: {course_row['course_name']} (Credits: {credits[i]}, Interest Points: {interest_points[i]})")
else:
    print(f'No optimal solution found. Status: {m.status}')