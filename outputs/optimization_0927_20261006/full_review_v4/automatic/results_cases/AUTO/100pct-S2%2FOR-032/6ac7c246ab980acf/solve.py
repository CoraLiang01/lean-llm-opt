import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == 'operations research'
ops_research_courses = df[ops_research_mask].copy()
if ops_research_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_courses['course_id'].tolist()
try:
    credits = dict(zip(course_ids, ops_research_courses['credits'].astype(int)))
    interest_points = dict(zip(course_ids, ops_research_courses['interest_points'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x_vars[c] for c in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x_vars[c] for c in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected_courses = [c for c in course_ids if x_vars[c].X > 0.5]
    total_credits = sum((credits[c] for c in selected_courses))
    total_interest = sum((interest_points[c] for c in selected_courses))
    print(f'Optimal total interest points: {m.objVal}')
    print(f'Total credits used: {total_credits} / 20')
    print(f'Selected Operations Research courses ({len(selected_courses)}):')
    for c in selected_courses:
        course_row = ops_research_courses[ops_research_courses['course_id'] == c].iloc[0]
        print(f"  {c}: {course_row['course_name']} | Credits: {credits[c]}, Interest Points: {interest_points[c]}")
else:
    print(f'No optimal solution found. Status: {m.status}')