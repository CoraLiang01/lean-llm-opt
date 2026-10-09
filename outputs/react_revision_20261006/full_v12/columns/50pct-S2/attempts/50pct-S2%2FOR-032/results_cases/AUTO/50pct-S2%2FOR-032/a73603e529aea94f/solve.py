CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an ‚Äúinterest points‚Äù score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['course_id', 'course_name', 'discipline', 'credits', 'interest_points'],
             'file_index': 0,
             'file_name': 'courses_42.csv',
             'filters': {'conditions': [{'column': 'discipline',
                                         'dtype': 'string',
                                         'evidence': '"subset of courses from the \'Operations Research\' discipline"',
                                         'operator': 'exact',
                                         'value': 'Operations Research'}],
                         'logic': 'and'},
             'original_rows': 42,
             'records': [{'source_row': 21,
                          'values': {'course_id': 'C22',
                                     'course_name': 'Operations Research: Linear Programming',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '95'}},
                         {'source_row': 22,
                          'values': {'course_id': 'C23',
                                     'course_name': 'Integer Programming',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '92'}},
                         {'source_row': 23,
                          'values': {'course_id': 'C24',
                                     'course_name': 'Stochastic Processes',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '86'}},
                         {'source_row': 24,
                          'values': {'course_id': 'C25',
                                     'course_name': 'Simulation Modeling',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '82'}},
                         {'source_row': 25,
                          'values': {'course_id': 'C26',
                                     'course_name': 'Network Flows',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '85'}},
                         {'source_row': 26,
                          'values': {'course_id': 'C27',
                                     'course_name': 'Queueing Theory',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '80'}},
                         {'source_row': 27,
                          'values': {'course_id': 'C28',
                                     'course_name': 'Revenue Management',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '88'}}],
             'returned_rows': 7,
             'role': 'course catalog',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    frame = CSVQA_FRAMES['file_0_view_0']
    or_mask = frame['discipline'].str.casefold() == 'operations research'
    or_courses = frame[or_mask]
    course_ids = []
    credits = {}
    interest_points = {}
    for (source_row, row) in or_courses.iterrows():
        course_id = row['course_id']
        course_ids.append(course_id)
        try:
            credits[course_id] = float(row['credits'])
        except Exception:
            raise ValueError(f'Invalid or missing credits for course {course_id}')
        try:
            interest_points[course_id] = float(row['interest_points'])
        except Exception:
            raise ValueError(f'Invalid or missing interest_points for course {course_id}')
    if len(course_ids) == 0:
        raise ValueError('No Operations Research courses found in the data.')
    if set(credits.keys()) != set(course_ids) or set(interest_points.keys()) != set(course_ids):
        raise ValueError('Missing credits or interest_points for some courses.')
    m = gp.Model('OR_Course_Selection')
    x_vars = m.addVars(course_ids, vtype=gp.GRB.BINARY, lb=0, name='')
    m.setObjective(gp.quicksum((interest_points[i] * x_vars[i] for i in course_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((credits[i] * x_vars[i] for i in course_ids)) <= 20, name='credit_cap')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Status: {m.status}')