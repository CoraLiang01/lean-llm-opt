import pandas as pd
import gurobipy as gp
from gurobipy import GRB
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
discipline_col = courses_df['discipline'].str.casefold().str.strip()
or_mask = discipline_col == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_course_ids = list(or_courses_df['course_id'])
try:
    credits = {cid: int(c) for (cid, c) in zip(or_courses_df['course_id'], or_courses_df['credits'])}
    interest_points = {cid: int(ip) for (cid, ip) in zip(or_courses_df['course_id'], or_courses_df['interest_points'])}
except Exception as e:
    raise ValueError(f'Error converting credits or interest_points to int: {e}')
if set(credits.keys()) != set(or_course_ids) or set(interest_points.keys()) != set(or_course_ids):
    raise ValueError('Mismatch in course_id keys for credits or interest_points.')
m = gp.Model('or_course_selection')
x_vars = m.addVars(or_course_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.addConstr(gp.quicksum((credits[cid] * x_vars[cid] for cid in or_course_ids)) <= 20, name='credit_limit')
m.setObjective(gp.quicksum((interest_points[cid] * x_vars[cid] for cid in or_course_ids)), GRB.MAXIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for cid in or_course_ids:
        print(f'{x_vars[cid].VarName} {x_vars[cid].X}')
else:
    print(f'Solver status: {m.Status}')