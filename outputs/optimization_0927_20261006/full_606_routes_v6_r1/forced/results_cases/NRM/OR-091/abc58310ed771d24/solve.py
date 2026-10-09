CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'You are given a catalog of 42 university courses across six disciplines: Literature, Mathematics, Physics, '
          'Operations Research, Computer Science, and Chemistry. Each course has an associated number of credits and '
          'an “interest points” score. All the data is contained in the file courses_42.csv. The student wants to '
          'enroll in a subset of courses from the "Operations Research" discipline. The goal is to choose a '
          'combination of Operations Research courses whose total credits no more than 20 and whose total interest '
          'points are maximized.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['course_id', 'course_name', 'discipline', 'credits', 'interest_points'],
             'file_index': 0,
             'file_name': 'courses_42.csv',
             'filters': {'conditions': [{'column': 'discipline',
                                         'dtype': 'string',
                                         'evidence': '"Operations Research"',
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
             'role': 'Operations Research courses with credits and interest points',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB
table = None
for t in CSVQA_DATA['tables']:
    if t['table_id'] == 'file_0_view_0':
        table = t
        break
if table is None:
    raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
records = []
for rec in table['records']:
    if rec['values']['discipline'] == 'Operations Research':
        records.append(rec)
if len(records) == 0:
    raise ValueError('No Operations Research courses found in data.')
C = []
a_c = {}
p_c = {}
for rec in records:
    cid = rec['values']['course_id']
    C.append(cid)
    try:
        a_c[cid] = int(rec['values']['credits'])
        p_c[cid] = int(rec['values']['interest_points'])
    except Exception as e:
        raise ValueError(f'Invalid numeric value in record {cid}: {e}')
if set(a_c.keys()) != set(C) or set(p_c.keys()) != set(C):
    raise ValueError('Parameter coverage mismatch for courses.')
m = gp.Model('OR_Course_Selection')
x_vars = m.addVars(C, vtype=GRB.BINARY, name='')
m.setObjective(gp.quicksum((p_c[c] * x_vars[c] for c in C)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((a_c[c] * x_vars[c] for c in C)) <= 20, name='credit_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')