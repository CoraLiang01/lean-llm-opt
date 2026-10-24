CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A design team must select exactly one package option from each component family. Each option has a value, '
          'weight, and labor-hour requirement in option_catalog.csv, while total resource limits are listed in '
          'resource_limits.csv.\n'
          '\n'
          'Formulate a maximum-value multi-choice knapsack model. For each family-option pair c-o, define x_co as a '
          'binary variable equal to 1 if option o is selected from family c. The objective is to maximize total '
          'selected value. The model should include exactly-one-option constraints for every family, total weight and '
          'labor-hour constraints, and binary restrictions for all option-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'LaborHours'],
             'file_index': 0,
             'file_name': 'option_catalog.csv',
             'filters': {},
             'original_rows': 18,
             'records': [{'source_row': 0,
                          'values': {'Family': 'C1', 'LaborHours': '6', 'Option': 'O1', 'Value': '18', 'Weight': '5'}},
                         {'source_row': 1,
                          'values': {'Family': 'C1', 'LaborHours': '9', 'Option': 'O2', 'Value': '27', 'Weight': '8'}},
                         {'source_row': 2,
                          'values': {'Family': 'C1',
                                     'LaborHours': '11',
                                     'Option': 'O3',
                                     'Value': '30',
                                     'Weight': '10'}},
                         {'source_row': 3,
                          'values': {'Family': 'C2', 'LaborHours': '8', 'Option': 'O1', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 4,
                          'values': {'Family': 'C2',
                                     'LaborHours': '13',
                                     'Option': 'O2',
                                     'Value': '32',
                                     'Weight': '11'}},
                         {'source_row': 5,
                          'values': {'Family': 'C2', 'LaborHours': '10', 'Option': 'O3', 'Value': '28', 'Weight': '9'}},
                         {'source_row': 6,
                          'values': {'Family': 'C3', 'LaborHours': '7', 'Option': 'O1', 'Value': '22', 'Weight': '6'}},
                         {'source_row': 7,
                          'values': {'Family': 'C3',
                                     'LaborHours': '14',
                                     'Option': 'O2',
                                     'Value': '35',
                                     'Weight': '12'}},
                         {'source_row': 8,
                          'values': {'Family': 'C3',
                                     'LaborHours': '12',
                                     'Option': 'O3',
                                     'Value': '31',
                                     'Weight': '10'}},
                         {'source_row': 9,
                          'values': {'Family': 'C4', 'LaborHours': '8', 'Option': 'O1', 'Value': '20', 'Weight': '5'}},
                         {'source_row': 10,
                          'values': {'Family': 'C4', 'LaborHours': '11', 'Option': 'O2', 'Value': '29', 'Weight': '9'}},
                         {'source_row': 11,
                          'values': {'Family': 'C4',
                                     'LaborHours': '13',
                                     'Option': 'O3',
                                     'Value': '34',
                                     'Weight': '12'}},
                         {'source_row': 12,
                          'values': {'Family': 'C5', 'LaborHours': '7', 'Option': 'O1', 'Value': '23', 'Weight': '7'}},
                         {'source_row': 13,
                          'values': {'Family': 'C5',
                                     'LaborHours': '12',
                                     'Option': 'O2',
                                     'Value': '33',
                                     'Weight': '11'}},
                         {'source_row': 14,
                          'values': {'Family': 'C5', 'LaborHours': '10', 'Option': 'O3', 'Value': '26', 'Weight': '8'}},
                         {'source_row': 15,
                          'values': {'Family': 'C6', 'LaborHours': '6', 'Option': 'O1', 'Value': '21', 'Weight': '6'}},
                         {'source_row': 16,
                          'values': {'Family': 'C6',
                                     'LaborHours': '11',
                                     'Option': 'O2',
                                     'Value': '30',
                                     'Weight': '10'}},
                         {'source_row': 17,
                          'values': {'Family': 'C6',
                                     'LaborHours': '15',
                                     'Option': 'O3',
                                     'Value': '36',
                                     'Weight': '13'}}],
             'returned_rows': 18,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Resource', 'Limit'],
             'file_index': 1,
             'file_name': 'resource_limits.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Limit': '55', 'Resource': 'Weight'}},
                         {'source_row': 1, 'values': {'Limit': '64', 'Resource': 'LaborHours'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    option_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    limit_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    families = set()
    options = set()
    v = {}
    w = {}
    l = {}
    for rec in option_table['records']:
        vals = rec['values']
        c = vals['Family']
        o = vals['Option']
        families.add(c)
        options.add(o)
        key = (c, o)
        try:
            v[key] = float(vals['Value'])
            w[key] = float(vals['Weight'])
            l[key] = float(vals['LaborHours'])
        except KeyError as e:
            raise ValueError(f'Missing field in option record: {e}')
    for c in families:
        for o in options:
            if (c, o) not in v:
                raise ValueError(f'Missing option {o} for family {c}')
    limits = {}
    for rec in limit_table['records']:
        vals = rec['values']
        res = vals['Resource']
        limits[res] = float(vals['Limit'])
    if 'Weight' not in limits or 'LaborHours' not in limits:
        raise ValueError('Missing resource limits')
    W = limits['Weight']
    L = limits['LaborHours']
    m = gurobipy.Model()
    m.setParam('MIPGap', 1e-06)
    keys = [(c, o) for c in families for o in options]
    x = m.addVars(keys, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((v[c, o] * x[c, o] for c in families for o in options)), GRB.MAXIMIZE)
    for c in families:
        m.addConstr(gurobipy.quicksum((x[c, o] for o in options)) == 1, name=f'OneOption_{c}')
    m.addConstr(gurobipy.quicksum((w[c, o] * x[c, o] for c in families for o in options)) <= W, name='WeightCap')
    m.addConstr(gurobipy.quicksum((l[c, o] * x[c, o] for c in families for o in options)) <= L, name='LaborCap')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()