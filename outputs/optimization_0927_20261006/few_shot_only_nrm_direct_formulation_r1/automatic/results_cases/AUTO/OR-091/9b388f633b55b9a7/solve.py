import gurobipy as gp
import pandas as pd
import numpy as np
import re
courses_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
courses_df = pd.read_csv(courses_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
or_mask = courses_df[discipline_col].str.strip().str.casefold() == 'operations research'
or_courses_df = courses_df[or_mask].copy()
if or_courses_df.shape[0] == 0:
    raise ValueError("No courses found in the 'Operations Research' discipline.")
course_id_col = 'course_id'
credits_col = 'credits'
interest_col = 'interest_points'
or_courses_df.set_index(course_id_col, inplace=True, drop=False)

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
credits_dict = safe_int(or_courses_df[credits_col], credits_col).to_dict()
interest_dict = safe_int(or_courses_df[interest_col], interest_col).to_dict()
I = list(or_courses_df.index)
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(I, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_dict[i] * x_vars[i] for i in I)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits_dict[i] * x_vars[i] for i in I)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in I if x_vars[i].X > 0.5]
    total_credits = sum((credits_dict[i] for i in selected))
    total_interest = sum((interest_dict[i] for i in selected))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total Interest Points)')
    print(f'Total credits used: {total_credits} / 20')
    print('\n--- Selected Operations Research Courses ---')
    for i in selected:
        cname = or_courses_df.loc[i, 'course_name']
        credits = credits_dict[i]
        interest = interest_dict[i]
        print(f'  {i}: {cname} | Credits: {credits} | Interest Points: {interest}')
else:
    print(f'No optimal solution found. Status: {m.status}')