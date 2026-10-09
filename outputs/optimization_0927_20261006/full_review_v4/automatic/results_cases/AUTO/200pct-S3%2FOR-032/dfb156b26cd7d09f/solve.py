import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
discipline_target = 'operations research'
ops_research_mask = df[discipline_col].str.strip().str.casefold() == discipline_target
ops_research_courses = df[ops_research_mask].copy()
if ops_research_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_courses['course_id'].tolist()
try:
    interest_points = dict(zip(course_ids, ops_research_courses['interest_points'].astype(int)))
    credits = dict(zip(course_ids, ops_research_courses['credits'].astype(int)))
except Exception as e:
    raise ValueError(f"Error converting 'interest_points' or 'credits' to int: {e}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total interest points: {m.objVal:.0f}')
    selected = [i for i in course_ids if x_vars[i].X > 0.5]
    total_credits = sum((credits[i] for i in selected))
    print(f'Total credits used: {total_credits}')
    print('Selected Operations Research courses:')
    for i in selected:
        cname = ops_research_courses.loc[ops_research_courses['course_id'] == i, 'course_name'].values[0]
        print(f'  {i}: {cname} (Credits: {credits[i]}, Interest Points: {interest_points[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')