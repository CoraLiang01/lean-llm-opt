CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A supermarket manager needs to select a variety of products to stock in different sections of the store. '
          'Particularly, the store has several sections, each with a display space limit provided in "capacity.csv." '
          'The predefined price and shelf space requirement of each product are detailed in "products.csv." The '
          'objective is to determine the optimal number of units of each product to stock in each section to maximize '
          'the total revenue, while ensuring that the total space used by the products in each section does not exceed '
          'the available capacity. The decision variables x_ij denote the number of units of product j to be placed in '
          'section i.The decision variables must be integers.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['SectionID', 'Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Capacity': '100', 'SectionID': '1'}},
                         {'source_row': 1, 'values': {'Capacity': '150', 'SectionID': '2'}},
                         {'source_row': 2, 'values': {'Capacity': '120', 'SectionID': '3'}},
                         {'source_row': 3, 'values': {'Capacity': '130', 'SectionID': '4'}},
                         {'source_row': 4, 'values': {'Capacity': '90', 'SectionID': '5'}},
                         {'source_row': 5, 'values': {'Capacity': '110', 'SectionID': '6'}},
                         {'source_row': 6, 'values': {'Capacity': '160', 'SectionID': '7'}},
                         {'source_row': 7, 'values': {'Capacity': '140', 'SectionID': '8'}}],
             'returned_rows': 8,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': '1', 'Value': '10', 'Weight': '2'}},
                         {'source_row': 1, 'values': {'ProductName': '2', 'Value': '15', 'Weight': '3'}},
                         {'source_row': 2, 'values': {'ProductName': '3', 'Value': '8', 'Weight': '1'}},
                         {'source_row': 3, 'values': {'ProductName': '4', 'Value': '12', 'Weight': '2'}},
                         {'source_row': 4, 'values': {'ProductName': '5', 'Value': '20', 'Weight': '4'}},
                         {'source_row': 5, 'values': {'ProductName': '6', 'Value': '25', 'Weight': '5'}},
                         {'source_row': 6, 'values': {'ProductName': '7', 'Value': '5', 'Weight': '1'}},
                         {'source_row': 7, 'values': {'ProductName': '8', 'Value': '30', 'Weight': '6'}},
                         {'source_row': 8, 'values': {'ProductName': '9', 'Value': '18', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': '10', 'Value': '22', 'Weight': '4'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    capacity_table = next((t for t in tables if t['file_name'] == 'capacity.csv'))
    products_table = next((t for t in tables if t['file_name'] == 'products.csv'))
    capacities = {}
    for rec in capacity_table['records']:
        sid = rec['values']['SectionID']
        cap = rec['values']['Capacity']
        try:
            capacities[sid] = int(cap)
        except ValueError:
            raise ValueError(f'Invalid capacity value for SectionID {sid}')
    values = {}
    weights = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        val = rec['values']['Value']
        wt = rec['values']['Weight']
        try:
            values[pname] = int(val)
            weights[pname] = int(wt)
        except ValueError:
            raise ValueError(f'Invalid value or weight for ProductName {pname}')
    I = list(capacities.keys())
    J = list(values.keys())
    for i in I:
        if i not in capacities:
            raise ValueError(f'Missing capacity for SectionID {i}')
    for j in J:
        if j not in values or j not in weights:
            raise ValueError(f'Missing value or weight for ProductName {j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.INTEGER, lb=0, name='')
    model.setObjective(gurobipy.quicksum((values[j] * x[i, j] for i in I for j in J)), GRB.MAXIMIZE)
    for i in I:
        model.addConstr(gurobipy.quicksum((weights[j] * x[i, j] for j in J)) <= capacities[i], name=f'cap_{i}')
    model.optimize()
    return model
m = solve_problem()