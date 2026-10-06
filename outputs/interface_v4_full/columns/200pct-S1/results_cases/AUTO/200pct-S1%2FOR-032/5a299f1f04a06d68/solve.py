import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',')
discipline_col = 'discipline'
ops_research_mask = courses_df[discipline_col].astype(str).str.casefold().str.strip() == 'operations research'
ops_research_courses = courses_df[ops_research_mask].copy()
if ops_research_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
C = list(ops_research_courses['course_id'])
credits = dict(zip(ops_research_courses['course_id'], ops_research_courses['credits']))
interest_points = dict(zip(ops_research_courses['course_id'], ops_research_courses['interest_points']))
for c in C:
    if c not in credits or c not in interest_points:
        raise ValueError(f'Missing credits or interest_points for course_id {c}')
m = gp.Model('OR_Course_Selection')
x = m.addVars(C, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[c] * x[c] for c in C)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[c] * x[c] for c in C)) <= 20, name='credit_limit')
m.optimize()