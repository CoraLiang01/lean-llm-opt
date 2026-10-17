CSVQA_DATA = {'ignored_file_indices': [],
 'query': "The company's goal is to select a subset of 110 potential investment projects that willmaximize the total "
          'expected Net Present Value (NPV).The optimization must satisfy the budget limit and the following strategic '
          'dependencies: (1)Budget Constraint:Total capital investment cannot exceed 1,000 k$; (2)Mutually Exclusive '
          'Constraint (Projects 4 & 7):Selectat most oneof Project 4 (R&D Initiative Alpha) or Project 7 (Global '
          'Expansion Pilot); (3)Pre-requisite Constraint (Project 6 requires 1):Project 6 (System Automation) can only '
          'be selected if Project 1(Infrastructure Upgrade) is also selected; (4)Contingent Constraint (Project 10 '
          'requires 5):Project 10 (Customer Experience Platform) canonly be selected if Project 5(Staff Trainig '
          'Program) is also selected. All required data is provided in project.csv.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Project ID', 'Capital (k$)', 'NPV (k$)'],
             'file_index': 0,
             'file_name': 'project.csv',
             'filters': {},
             'original_rows': 110,
             'records': [{'source_row': 0, 'values': {'Capital (k$)': '50', 'NPV (k$)': '60', 'Project ID': '1'}},
                         {'source_row': 1, 'values': {'Capital (k$)': '40', 'NPV (k$)': '50', 'Project ID': '2'}},
                         {'source_row': 2, 'values': {'Capital (k$)': '30', 'NPV (k$)': '45', 'Project ID': '3'}},
                         {'source_row': 3, 'values': {'Capital (k$)': '25', 'NPV (k$)': '35', 'Project ID': '4'}},
                         {'source_row': 4, 'values': {'Capital (k$)': '20', 'NPV (k$)': '28', 'Project ID': '5'}},
                         {'source_row': 5, 'values': {'Capital (k$)': '65', 'NPV (k$)': '75', 'Project ID': '6'}},
                         {'source_row': 6, 'values': {'Capital (k$)': '80', 'NPV (k$)': '100', 'Project ID': '7'}},
                         {'source_row': 7, 'values': {'Capital (k$)': '15', 'NPV (k$)': '20', 'Project ID': '8'}},
                         {'source_row': 8, 'values': {'Capital (k$)': '48', 'NPV (k$)': '62', 'Project ID': '9'}},
                         {'source_row': 9, 'values': {'Capital (k$)': '55', 'NPV (k$)': '70', 'Project ID': '10'}},
                         {'source_row': 10, 'values': {'Capital (k$)': '87', 'NPV (k$)': '117', 'Project ID': '11'}},
                         {'source_row': 11, 'values': {'Capital (k$)': '82', 'NPV (k$)': '109', 'Project ID': '12'}},
                         {'source_row': 12, 'values': {'Capital (k$)': '123', 'NPV (k$)': '172', 'Project ID': '13'}},
                         {'source_row': 13, 'values': {'Capital (k$)': '137', 'NPV (k$)': '180', 'Project ID': '14'}},
                         {'source_row': 14, 'values': {'Capital (k$)': '43', 'NPV (k$)': '62', 'Project ID': '15'}},
                         {'source_row': 15, 'values': {'Capital (k$)': '110', 'NPV (k$)': '147', 'Project ID': '16'}},
                         {'source_row': 16, 'values': {'Capital (k$)': '94', 'NPV (k$)': '141', 'Project ID': '17'}},
                         {'source_row': 17, 'values': {'Capital (k$)': '101', 'NPV (k$)': '151', 'Project ID': '18'}},
                         {'source_row': 18, 'values': {'Capital (k$)': '54', 'NPV (k$)': '75', 'Project ID': '19'}},
                         {'source_row': 19, 'values': {'Capital (k$)': '143', 'NPV (k$)': '185', 'Project ID': '20'}},
                         {'source_row': 20, 'values': {'Capital (k$)': '37', 'NPV (k$)': '51', 'Project ID': '21'}},
                         {'source_row': 21, 'values': {'Capital (k$)': '46', 'NPV (k$)': '59', 'Project ID': '22'}},
                         {'source_row': 22, 'values': {'Capital (k$)': '81', 'NPV (k$)': '105', 'Project ID': '23'}},
                         {'source_row': 23, 'values': {'Capital (k$)': '100', 'NPV (k$)': '123', 'Project ID': '24'}},
                         {'source_row': 24, 'values': {'Capital (k$)': '80', 'NPV (k$)': '104', 'Project ID': '25'}},
                         {'source_row': 25, 'values': {'Capital (k$)': '102', 'NPV (k$)': '153', 'Project ID': '26'}},
                         {'source_row': 26, 'values': {'Capital (k$)': '109', 'NPV (k$)': '163', 'Project ID': '27'}},
                         {'source_row': 27, 'values': {'Capital (k$)': '79', 'NPV (k$)': '114', 'Project ID': '28'}},
                         {'source_row': 28, 'values': {'Capital (k$)': '100', 'NPV (k$)': '125', 'Project ID': '29'}},
                         {'source_row': 29, 'values': {'Capital (k$)': '119', 'NPV (k$)': '160', 'Project ID': '30'}},
                         {'source_row': 30, 'values': {'Capital (k$)': '139', 'NPV (k$)': '166', 'Project ID': '31'}},
                         {'source_row': 31, 'values': {'Capital (k$)': '11', 'NPV (k$)': '15', 'Project ID': '32'}},
                         {'source_row': 32, 'values': {'Capital (k$)': '144', 'NPV (k$)': '184', 'Project ID': '33'}},
                         {'source_row': 33, 'values': {'Capital (k$)': '56', 'NPV (k$)': '86', 'Project ID': '34'}},
                         {'source_row': 34, 'values': {'Capital (k$)': '59', 'NPV (k$)': '85', 'Project ID': '35'}},
                         {'source_row': 35, 'values': {'Capital (k$)': '73', 'NPV (k$)': '109', 'Project ID': '36'}},
                         {'source_row': 36, 'values': {'Capital (k$)': '136', 'NPV (k$)': '187', 'Project ID': '37'}},
                         {'source_row': 37, 'values': {'Capital (k$)': '85', 'NPV (k$)': '115', 'Project ID': '38'}},
                         {'source_row': 38, 'values': {'Capital (k$)': '111', 'NPV (k$)': '155', 'Project ID': '39'}},
                         {'source_row': 39, 'values': {'Capital (k$)': '103', 'NPV (k$)': '137', 'Project ID': '40'}},
                         {'source_row': 40, 'values': {'Capital (k$)': '146', 'NPV (k$)': '204', 'Project ID': '41'}},
                         {'source_row': 41, 'values': {'Capital (k$)': '82', 'NPV (k$)': '98', 'Project ID': '42'}},
                         {'source_row': 42, 'values': {'Capital (k$)': '144', 'NPV (k$)': '189', 'Project ID': '43'}},
                         {'source_row': 43, 'values': {'Capital (k$)': '143', 'NPV (k$)': '178', 'Project ID': '44'}},
                         {'source_row': 44, 'values': {'Capital (k$)': '85', 'NPV (k$)': '130', 'Project ID': '45'}},
                         {'source_row': 45, 'values': {'Capital (k$)': '121', 'NPV (k$)': '148', 'Project ID': '46'}},
                         {'source_row': 46, 'values': {'Capital (k$)': '147', 'NPV (k$)': '205', 'Project ID': '47'}},
                         {'source_row': 47, 'values': {'Capital (k$)': '92', 'NPV (k$)': '128', 'Project ID': '48'}},
                         {'source_row': 48, 'values': {'Capital (k$)': '115', 'NPV (k$)': '165', 'Project ID': '49'}},
                         {'source_row': 49, 'values': {'Capital (k$)': '13', 'NPV (k$)': '18', 'Project ID': '50'}},
                         {'source_row': 50, 'values': {'Capital (k$)': '145', 'NPV (k$)': '203', 'Project ID': '51'}},
                         {'source_row': 51, 'values': {'Capital (k$)': '48', 'NPV (k$)': '72', 'Project ID': '52'}},
                         {'source_row': 52, 'values': {'Capital (k$)': '74', 'NPV (k$)': '92', 'Project ID': '53'}},
                         {'source_row': 53, 'values': {'Capital (k$)': '135', 'NPV (k$)': '167', 'Project ID': '54'}},
                         {'source_row': 54, 'values': {'Capital (k$)': '28', 'NPV (k$)': '40', 'Project ID': '55'}},
                         {'source_row': 55, 'values': {'Capital (k$)': '74', 'NPV (k$)': '89', 'Project ID': '56'}},
                         {'source_row': 56, 'values': {'Capital (k$)': '89', 'NPV (k$)': '111', 'Project ID': '57'}},
                         {'source_row': 57, 'values': {'Capital (k$)': '24', 'NPV (k$)': '38', 'Project ID': '58'}},
                         {'source_row': 58, 'values': {'Capital (k$)': '141', 'NPV (k$)': '190', 'Project ID': '59'}},
                         {'source_row': 59, 'values': {'Capital (k$)': '27', 'NPV (k$)': '33', 'Project ID': '60'}},
                         {'source_row': 60, 'values': {'Capital (k$)': '106', 'NPV (k$)': '148', 'Project ID': '61'}},
                         {'source_row': 61, 'values': {'Capital (k$)': '37', 'NPV (k$)': '44', 'Project ID': '62'}},
                         {'source_row': 62, 'values': {'Capital (k$)': '114', 'NPV (k$)': '167', 'Project ID': '63'}},
                         {'source_row': 63, 'values': {'Capital (k$)': '11', 'NPV (k$)': '14', 'Project ID': '64'}},
                         {'source_row': 64, 'values': {'Capital (k$)': '113', 'NPV (k$)': '153', 'Project ID': '65'}},
                         {'source_row': 65, 'values': {'Capital (k$)': '61', 'NPV (k$)': '91', 'Project ID': '66'}},
                         {'source_row': 66, 'values': {'Capital (k$)': '62', 'NPV (k$)': '94', 'Project ID': '67'}},
                         {'source_row': 67, 'values': {'Capital (k$)': '145', 'NPV (k$)': '188', 'Project ID': '68'}},
                         {'source_row': 68, 'values': {'Capital (k$)': '148', 'NPV (k$)': '177', 'Project ID': '69'}},
                         {'source_row': 69, 'values': {'Capital (k$)': '30', 'NPV (k$)': '36', 'Project ID': '70'}},
                         {'source_row': 70, 'values': {'Capital (k$)': '134', 'NPV (k$)': '179', 'Project ID': '71'}},
                         {'source_row': 71, 'values': {'Capital (k$)': '97', 'NPV (k$)': '143', 'Project ID': '72'}},
                         {'source_row': 72, 'values': {'Capital (k$)': '103', 'NPV (k$)': '123', 'Project ID': '73'}},
                         {'source_row': 73, 'values': {'Capital (k$)': '102', 'NPV (k$)': '126', 'Project ID': '74'}},
                         {'source_row': 74, 'values': {'Capital (k$)': '141', 'NPV (k$)': '173', 'Project ID': '75'}},
                         {'source_row': 75, 'values': {'Capital (k$)': '109', 'NPV (k$)': '152', 'Project ID': '76'}},
                         {'source_row': 76, 'values': {'Capital (k$)': '118', 'NPV (k$)': '170', 'Project ID': '77'}},
                         {'source_row': 77, 'values': {'Capital (k$)': '93', 'NPV (k$)': '148', 'Project ID': '78'}},
                         {'source_row': 78, 'values': {'Capital (k$)': '95', 'NPV (k$)': '142', 'Project ID': '79'}},
                         {'source_row': 79, 'values': {'Capital (k$)': '92', 'NPV (k$)': '118', 'Project ID': '80'}},
                         {'source_row': 80, 'values': {'Capital (k$)': '71', 'NPV (k$)': '104', 'Project ID': '81'}},
                         {'source_row': 81, 'values': {'Capital (k$)': '127', 'NPV (k$)': '159', 'Project ID': '82'}},
                         {'source_row': 82, 'values': {'Capital (k$)': '126', 'NPV (k$)': '163', 'Project ID': '83'}},
                         {'source_row': 83, 'values': {'Capital (k$)': '123', 'NPV (k$)': '165', 'Project ID': '84'}},
                         {'source_row': 84, 'values': {'Capital (k$)': '137', 'NPV (k$)': '164', 'Project ID': '85'}},
                         {'source_row': 85, 'values': {'Capital (k$)': '124', 'NPV (k$)': '149', 'Project ID': '86'}},
                         {'source_row': 86, 'values': {'Capital (k$)': '103', 'NPV (k$)': '144', 'Project ID': '87'}},
                         {'source_row': 87, 'values': {'Capital (k$)': '119', 'NPV (k$)': '166', 'Project ID': '88'}},
                         {'source_row': 88, 'values': {'Capital (k$)': '87', 'NPV (k$)': '105', 'Project ID': '89'}},
                         {'source_row': 89, 'values': {'Capital (k$)': '87', 'NPV (k$)': '130', 'Project ID': '90'}},
                         {'source_row': 90, 'values': {'Capital (k$)': '92', 'NPV (k$)': '138', 'Project ID': '91'}},
                         {'source_row': 91, 'values': {'Capital (k$)': '69', 'NPV (k$)': '96', 'Project ID': '92'}},
                         {'source_row': 92, 'values': {'Capital (k$)': '149', 'NPV (k$)': '178', 'Project ID': '93'}},
                         {'source_row': 93, 'values': {'Capital (k$)': '146', 'NPV (k$)': '193', 'Project ID': '94'}},
                         {'source_row': 94, 'values': {'Capital (k$)': '47', 'NPV (k$)': '68', 'Project ID': '95'}},
                         {'source_row': 95, 'values': {'Capital (k$)': '12', 'NPV (k$)': '18', 'Project ID': '96'}},
                         {'source_row': 96, 'values': {'Capital (k$)': '101', 'NPV (k$)': '131', 'Project ID': '97'}},
                         {'source_row': 97, 'values': {'Capital (k$)': '69', 'NPV (k$)': '103', 'Project ID': '98'}},
                         {'source_row': 98, 'values': {'Capital (k$)': '46', 'NPV (k$)': '65', 'Project ID': '99'}},
                         {'source_row': 99, 'values': {'Capital (k$)': '79', 'NPV (k$)': '108', 'Project ID': '100'}},
                         {'source_row': 100, 'values': {'Capital (k$)': '93', 'NPV (k$)': '130', 'Project ID': '101'}},
                         {'source_row': 101, 'values': {'Capital (k$)': '49', 'NPV (k$)': '60', 'Project ID': '102'}},
                         {'source_row': 102, 'values': {'Capital (k$)': '110', 'NPV (k$)': '157', 'Project ID': '103'}},
                         {'source_row': 103, 'values': {'Capital (k$)': '133', 'NPV (k$)': '184', 'Project ID': '104'}},
                         {'source_row': 104, 'values': {'Capital (k$)': '92', 'NPV (k$)': '130', 'Project ID': '105'}},
                         {'source_row': 105, 'values': {'Capital (k$)': '106', 'NPV (k$)': '128', 'Project ID': '106'}},
                         {'source_row': 106, 'values': {'Capital (k$)': '144', 'NPV (k$)': '201', 'Project ID': '107'}},
                         {'source_row': 107, 'values': {'Capital (k$)': '124', 'NPV (k$)': '173', 'Project ID': '108'}},
                         {'source_row': 108, 'values': {'Capital (k$)': '148', 'NPV (k$)': '189', 'Project ID': '109'}},
                         {'source_row': 109,
                          'values': {'Capital (k$)': '127', 'NPV (k$)': '177', 'Project ID': '110'}}],
             'returned_rows': 110,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    project_ids = []
    c = {}
    n = {}
    for rec in records:
        vals = rec['values']
        pid = vals['Project ID']
        if 'Capital (k$)' not in vals or 'NPV (k$)' not in vals:
            raise ValueError(f'Missing data for project {pid}')
        try:
            capital = float(vals['Capital (k$)'])
            npv = float(vals['NPV (k$)'])
        except ValueError as e:
            raise ValueError(f'Invalid numeric value for project {pid}: {e}')
        project_ids.append(pid)
        c[pid] = capital
        n[pid] = npv
    if len(project_ids) != len(c) or len(project_ids) != len(n):
        raise ValueError('Inconsistent project data')
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    y = m.addVars(project_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((n[pid] * y[pid] for pid in project_ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((c[pid] * y[pid] for pid in project_ids)) <= 1000, name='budget')
    m.addConstr(y['4'] + y['7'] <= 1, name='mutually_exclusive')
    m.addConstr(y['6'] <= y['1'], name='pre_requisite')
    m.addConstr(y['10'] <= y['5'], name='contingent')
    m.optimize()
    return m
m = solve_problem()