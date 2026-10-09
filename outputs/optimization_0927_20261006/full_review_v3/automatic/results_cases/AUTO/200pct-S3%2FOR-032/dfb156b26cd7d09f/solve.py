import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
discipline_col = 'discipline'
discipline_norm = df[discipline_col].str.strip().str.casefold()
or_mask = discipline_norm == 'operations research'
df_or = df[or_mask].copy()
if df_or.empty:
    raise ValueError("No courses found with discipline == 'Operations Research'.")
course_ids = df_or['course_id'].tolist()

def extract_int_column(df, col, ids):
    vals = df[col].astype(str).str.strip()
    try:
        vals_int = vals.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{col}' contains non-integer values for Operations Research courses.") from e
    return dict(zip(ids, vals_int))
interest_points = extract_int_column(df_or, 'interest_points', course_ids)
credits = extract_int_column(df_or, 'credits', course_ids)
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_interest = sum((interest_points[i] for i in course_ids if x_vars[i].X > 0.5))
    total_credits = sum((credits[i] for i in course_ids if x_vars[i].X > 0.5))
    print(f'Optimal total interest points: {m.objVal:.0f}')
    print(f'Total credits used: {total_credits} / 20')
    print('--- Selected Operations Research Courses ---')
    for i in course_ids:
        if x_vars[i].X > 0.5:
            cname = df_or.loc[df_or['course_id'] == i, 'course_name'].values[0]
            print(f'  {i}: {cname} | Credits: {credits[i]}, Interest Points: {interest_points[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')