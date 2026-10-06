import gurobipy as gp
import pandas as pd
import numpy as np
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(courses_path, sep=',')
or_mask = df['discipline'].str.casefold().str.strip() == 'operations research'
or_courses = df.loc[or_mask].copy()
if or_courses.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_ids = list(or_courses['course_id'])
credits = dict(zip(or_courses['course_id'], or_courses['credits']))
interest_points = dict(zip(or_courses['course_id'], or_courses['interest_points']))
if set(course_ids) != set(credits.keys()) or set(course_ids) != set(interest_points.keys()):
    raise ValueError('Mismatch in course_ids and parameter keys for credits or interest_points.')

def solve_problem(course_ids, credits, interest_points):
    m = gp.Model('OR_Course_Selection')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x[i] for i in course_ids)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_problem(course_ids, credits, interest_points)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')