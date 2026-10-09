import pandas as pd
from gurobipy import Model, GRB, quicksum
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
discipline_norm = courses_df['discipline'].str.strip().str.casefold()
or_mask = discipline_norm == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in 'Operations Research' discipline.")
or_courses_df.set_index('course_id', inplace=True)
C = list(or_courses_df.index)
credits = or_courses_df['credits'].astype(int).to_dict()
interest_points = or_courses_df['interest_points'].astype(int).to_dict()
if set(credits.keys()) != set(C) or set(interest_points.keys()) != set(C):
    raise ValueError('Parameter keys do not match course index set.')
m = Model('or_course_selection')
x_vars = m.addVars(C, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(quicksum((interest_points[c] * x_vars[c] for c in C)), GRB.MAXIMIZE)
m.addConstr(quicksum((credits[c] * x_vars[c] for c in C)) <= 20, name='credit_limit')
m.optimize()