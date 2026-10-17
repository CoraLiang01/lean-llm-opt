CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['course_id', 'credits', 'interest_points', 'discipline'],
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
    I = []
    c = {}
    p = {}
    for table in CSVQA_DATA['tables']:
        if table.get('table_id') == 'file_0_view_0':
            for record in table['records']:
                values = record['values']
                course_id = values['course_id']
                credits = int(values['credits'])
                interest = int(values['interest_points'])
                I.append(course_id)
                c[course_id] = credits
                p[course_id] = interest
    if not set(I).issubset(c.keys()) or not set(I).issubset(p.keys()):
        missing = set(I) - set(c.keys()) | set(I) - set(p.keys())
        raise ValueError(f'Missing parameters for courses: {missing}')
    m = gurobipy.Model('OperationsResearchCourseSelection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((p[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((c[i] * x[i] for i in I)) <= 20, name='credit_limit')
    m.optimize()
    return m
m = solve_problem()