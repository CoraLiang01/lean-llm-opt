import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture11/courses_42.csv', sep=',', dtype=str, keep_default_na=False)
    or_mask = df['discipline'].str.casefold() == 'operations research'.casefold()
    or_courses_df = df[or_mask].copy()
    required_cols = ['course_id', 'credits', 'interest_points']
    for col in required_cols:
        if col not in or_courses_df.columns:
            raise KeyError(f'Missing required column: {col}')
    course_ids = list(or_courses_df['course_id'])
    if len(course_ids) == 0:
        raise ValueError('No Operations Research courses found in the data.')
    try:
        credits = pd.to_numeric(or_courses_df['credits'], errors='raise')
        interest_points = pd.to_numeric(or_courses_df['interest_points'], errors='raise')
    except Exception as e:
        raise ValueError(f'Failed to convert credits or interest_points to numeric: {e}')
    credits_dict = dict(zip(course_ids, credits))
    interest_points_dict = dict(zip(course_ids, interest_points))
    if set(credits_dict.keys()) != set(course_ids) or set(interest_points_dict.keys()) != set(course_ids):
        raise ValueError('Mismatch in parameter keys for Operations Research courses.')
    m = gp.Model('OR_Course_Selection')
    x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((interest_points_dict[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits_dict[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in course_ids:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()