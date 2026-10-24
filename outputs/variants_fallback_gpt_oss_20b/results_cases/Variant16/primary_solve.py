CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A product team must select exactly one design option from each component family. Option value, weight, and '
          'budget use are listed in option_catalog.csv, and total resource limits are listed in resource_limits.csv.\n'
          '\n'
          'Formulate a maximum-value multi-choice knapsack model. For each family-option pair g-o, define x_go as a '
          'binary variable equal to 1 if option o is selected from family g. The objective is to maximize total '
          'selected value. The model should include exactly-one-option constraints for every family, total weight and '
          'budget-use constraints, and binary restrictions for all option-selection variables.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['Family', 'Option', 'Value', 'Weight', 'BudgetUse'],
             'file_index': 0,
             'file_name': 'option_catalog.csv',
             'filters': {},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'BudgetUse': '8', 'Family': 'G1', 'Option': 'O1', 'Value': '22', 'Weight': '6'}},
                         {'source_row': 1,
                          'values': {'BudgetUse': '11', 'Family': 'G1', 'Option': 'O2', 'Value': '29', 'Weight': '9'}},
                         {'source_row': 2,
                          'values': {'BudgetUse': '13', 'Family': 'G1', 'Option': 'O3', 'Value': '31', 'Weight': '10'}},
                         {'source_row': 3,
                          'values': {'BudgetUse': '9', 'Family': 'G1', 'Option': 'O4', 'Value': '25', 'Weight': '7'}},
                         {'source_row': 4,
                          'values': {'BudgetUse': '8', 'Family': 'G2', 'Option': 'O1', 'Value': '24', 'Weight': '7'}},
                         {'source_row': 5,
                          'values': {'BudgetUse': '13', 'Family': 'G2', 'Option': 'O2', 'Value': '33', 'Weight': '11'}},
                         {'source_row': 6,
                          'values': {'BudgetUse': '10', 'Family': 'G2', 'Option': 'O3', 'Value': '28', 'Weight': '8'}},
                         {'source_row': 7,
                          'values': {'BudgetUse': '14', 'Family': 'G2', 'Option': 'O4', 'Value': '35', 'Weight': '12'}},
                         {'source_row': 8,
                          'values': {'BudgetUse': '12', 'Family': 'G3', 'Option': 'O1', 'Value': '30', 'Weight': '9'}},
                         {'source_row': 9,
                          'values': {'BudgetUse': '9', 'Family': 'G3', 'Option': 'O2', 'Value': '26', 'Weight': '7'}},
                         {'source_row': 10,
                          'values': {'BudgetUse': '16', 'Family': 'G3', 'Option': 'O3', 'Value': '38', 'Weight': '13'}},
                         {'source_row': 11,
                          'values': {'BudgetUse': '13', 'Family': 'G3', 'Option': 'O4', 'Value': '34', 'Weight': '11'}},
                         {'source_row': 12,
                          'values': {'BudgetUse': '7', 'Family': 'G4', 'Option': 'O1', 'Value': '21', 'Weight': '5'}},
                         {'source_row': 13,
                          'values': {'BudgetUse': '12', 'Family': 'G4', 'Option': 'O2', 'Value': '32', 'Weight': '10'}},
                         {'source_row': 14,
                          'values': {'BudgetUse': '15', 'Family': 'G4', 'Option': 'O3', 'Value': '36', 'Weight': '12'}},
                         {'source_row': 15,
                          'values': {'BudgetUse': '10', 'Family': 'G4', 'Option': 'O4', 'Value': '27', 'Weight': '8'}},
                         {'source_row': 16,
                          'values': {'BudgetUse': '11', 'Family': 'G5', 'Option': 'O1', 'Value': '29', 'Weight': '8'}},
                         {'source_row': 17,
                          'values': {'BudgetUse': '15', 'Family': 'G5', 'Option': 'O2', 'Value': '37', 'Weight': '12'}},
                         {'source_row': 18,
                          'values': {'BudgetUse': '12', 'Family': 'G5', 'Option': 'O3', 'Value': '33', 'Weight': '10'}},
                         {'source_row': 19,
                          'values': {'BudgetUse': '8', 'Family': 'G5', 'Option': 'O4', 'Value': '24', 'Weight': '6'}}],
             'returned_rows': 20,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['Resource', 'Limit'],
             'file_index': 1,
             'file_name': 'resource_limits.csv',
             'filters': {},
             'original_rows': 2,
             'records': [{'source_row': 0, 'values': {'Limit': '48', 'Resource': 'Weight'}},
                         {'source_row': 1, 'values': {'Limit': '60', 'Resource': 'BudgetUse'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    records0 = table0['records']
    families = set()
    options_of_family = {}
    v = {}
    w = {}
    b = {}
    keys = []
    for rec in records0:
        vals = rec['values']
        fam = vals['Family']
        opt = vals['Option']
        families.add(fam)
        options_of_family.setdefault(fam, set()).add(opt)
        key = (fam, opt)
        keys.append(key)
        try:
            v[key] = float(vals['Value'])
            w[key] = float(vals['Weight'])
            b[key] = float(vals['BudgetUse'])
        except KeyError as e:
            raise ValueError(f'Missing required column in option catalog: {e}')
    for fam in families:
        if fam not in options_of_family or not options_of_family[fam]:
            raise ValueError(f'Family {fam} has no options')
    records1 = table1['records']
    Wmax = None
    Bmax = None
    for rec in records1:
        vals = rec['values']
        res = vals['Resource']
        limit = float(vals['Limit'])
        if res == 'Weight':
            Wmax = limit
        elif res == 'BudgetUse':
            Bmax = limit
    if Wmax is None or Bmax is None:
        raise ValueError('Resource limits for Weight or BudgetUse missing')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(keys, vtype=GRB.BINARY, name='')
    model.setObjective(gurobipy.quicksum((v[key] * x[key] for key in keys)), GRB.MAXIMIZE)
    for fam in families:
        model.addConstr(gurobipy.quicksum((x[fam, opt] for opt in options_of_family[fam])) == 1, name=f'one_option_{fam}')
    model.addConstr(gurobipy.quicksum((w[key] * x[key] for key in keys)) <= Wmax, name='weight_capacity')
    model.addConstr(gurobipy.quicksum((b[key] * x[key] for key in keys)) <= Bmax, name='budget_capacity')
    model.optimize()
    return model
m = solve_problem()