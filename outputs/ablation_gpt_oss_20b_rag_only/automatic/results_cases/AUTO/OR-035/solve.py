CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['Capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0, 'values': {'ProductName': 'Baguette', 'Value': '888', 'Weight': '4'}},
                         {'source_row': 1, 'values': {'ProductName': 'Croissant', 'Value': '134', 'Weight': '2'}},
                         {'source_row': 2, 'values': {'ProductName': 'Sourdough', 'Value': '129', 'Weight': '4'}},
                         {'source_row': 3, 'values': {'ProductName': 'Rye Bread', 'Value': '370', 'Weight': '3'}},
                         {'source_row': 4, 'values': {'ProductName': 'Brioche', 'Value': '921', 'Weight': '2'}},
                         {'source_row': 5, 'values': {'ProductName': 'Focaccia', 'Value': '765', 'Weight': '1'}},
                         {'source_row': 6, 'values': {'ProductName': 'Ciabatta', 'Value': '154', 'Weight': '2'}},
                         {'source_row': 7, 'values': {'ProductName': 'Pita', 'Value': '837', 'Weight': '1'}},
                         {'source_row': 8, 'values': {'ProductName': 'Bagel', 'Value': '584', 'Weight': '3'}},
                         {'source_row': 9, 'values': {'ProductName': 'English Muffin', 'Value': '365', 'Weight': '3'}}],
             'returned_rows': 10,
             'role': 'descriptive non-unique role',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    prod_table = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    prod_records = prod_table['records']
    cap_table = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    cap_records = cap_table['records']
    P = []
    Profit = {}
    Weight = {}
    for rec in prod_records:
        values = rec['values']
        p = values['ProductName']
        P.append(p)
        try:
            Profit[p] = float(values['Value'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Profit for product {p}')
        try:
            Weight[p] = float(values['Weight'])
        except (KeyError, ValueError):
            raise ValueError(f'Missing or invalid Weight for product {p}')
    for p in P:
        if p not in Profit:
            raise ValueError(f'Profit missing for product {p}')
        if p not in Weight:
            raise ValueError(f'Weight missing for product {p}')
    if not cap_records:
        raise ValueError('Capacity record missing')
    cap_values = cap_records[0]['values']
    try:
        Capacity = float(cap_values['Capacity'])
    except (KeyError, ValueError):
        raise ValueError('Missing or invalid Capacity value')
    m = gp.Model()
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(P, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((Profit[p] * x[p] for p in P)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((Weight[p] * x[p] for p in P)) <= Capacity)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for var in m.getVars():
            print(var.VarName, var.X)
    else:
        print(m.Status)
    return m
m = solve_problem()