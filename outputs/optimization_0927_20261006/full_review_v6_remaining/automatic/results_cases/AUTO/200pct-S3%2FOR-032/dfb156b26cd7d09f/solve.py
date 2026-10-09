import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
discipline_norm = courses_df[discipline_col].str.strip().str.casefold()
ops_research_mask = discipline_norm == 'operations research'
ops_research_courses_df = courses_df[ops_research_mask].copy()
if ops_research_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = ops_research_courses_df['course_id'].tolist()
try:
    interest_points = ops_research_courses_df.set_index('course_id')['interest_points'].astype(int).to_dict()
    credits = ops_research_courses_df.set_index('course_id')['credits'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'interest_points' or 'credits' to int: {e}")
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x_vars[c] for c in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x_vars[c] for c in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[c] for c in course_ids if x_vars[c].X > 0.5))
    total_credits = sum((credits[c] for c in course_ids if x_vars[c].X > 0.5))
    print(f'Optimal total interest points: {total_interest}')
    print(f'Total credits used: {total_credits}')
    print('\nSelected Operations Research courses:')
    for c in course_ids:
        if x_vars[c].X > 0.5:
            course_name = ops_research_courses_df.loc[ops_research_courses_df['course_id'] == c, 'course_name'].values[0]
            print(f'  {c}: {course_name} (Credits: {credits[c]}, Interest Points: {interest_points[c]})')
else:
    print(f'No optimal solution found. Status: {m.status}')