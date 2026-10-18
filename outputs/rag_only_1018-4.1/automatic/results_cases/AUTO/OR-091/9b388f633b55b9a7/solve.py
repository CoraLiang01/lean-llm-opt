import pandas as pd
from gurobipy import Model, GRB, quicksum
courses_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', dtype=str, keep_default_na=False)
discipline_norm = courses_df['discipline'].str.strip().str.casefold()
or_mask = discipline_norm == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.empty:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
or_courses_df.set_index('course_id', inplace=True)
I = list(or_courses_df.index)
credits = or_courses_df['credits'].astype(int).to_dict()
interest_points = or_courses_df['interest_points'].astype(int).to_dict()
for i in I:
    if i not in credits or i not in interest_points:
        raise ValueError(f'Missing data for course_id {i}')
m = Model('or_course_selection')
x_vars = m.addVars(I, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(quicksum((interest_points[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
m.addConstr(quicksum((credits[i] * x_vars[i] for i in I)) <= 20, name='credit_limit')
m.optimize()