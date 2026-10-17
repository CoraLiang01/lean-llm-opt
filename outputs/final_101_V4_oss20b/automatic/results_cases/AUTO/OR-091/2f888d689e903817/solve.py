CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['course_id', 'discipline', 'credits', 'interest_points'],
             'file_index': 0,
             'file_name': 'courses_42.csv',
             'filters': {'conditions': [{'column': 'discipline',
                                         'dtype': 'string',
                                         'evidence': 'Operations Research',
                                         'operator': 'eq',
                                         'value': 'Operations Research'}],
                         'logic': 'and'},
             'original_rows': 42,
             'records': [{'source_row': 21,
                          'values': {'course_id': 'C22',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '95'}},
                         {'source_row': 22,
                          'values': {'course_id': 'C23',
                                     'credits': '5',
                                     'discipline': 'Operations Research',
                                     'interest_points': '92'}},
                         {'source_row': 23,
                          'values': {'course_id': 'C24',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '86'}},
                         {'source_row': 24,
                          'values': {'course_id': 'C25',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '82'}},
                         {'source_row': 25,
                          'values': {'course_id': 'C26',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '85'}},
                         {'source_row': 26,
                          'values': {'course_id': 'C27',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '80'}},
                         {'source_row': 27,
                          'values': {'course_id': 'C28',
                                     'credits': '4',
                                     'discipline': 'Operations Research',
                                     'interest_points': '88'}}],
             'returned_rows': 7,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    ids = []
    c = {}
    p = {}
    for rec in table['records']:
        values = rec['values']
        if values.get('discipline') != 'Operations Research':
            continue
        course_id = values['course_id']
        ids.append(course_id)
        try:
            c_val = int(values['credits'])
            p_val = int(values['interest_points'])
        except (KeyError, ValueError) as e:
            raise ValueError(f'Missing or invalid parameter for course {course_id}') from e
        c[course_id] = c_val
        p[course_id] = p_val
    if not ids:
        raise ValueError('No Operations Research courses found in data')
    for cid in ids:
        if cid not in c or cid not in p:
            raise ValueError(f'Parameter missing for course {cid}')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((p[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((c[i] * x[i] for i in ids)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_problem()