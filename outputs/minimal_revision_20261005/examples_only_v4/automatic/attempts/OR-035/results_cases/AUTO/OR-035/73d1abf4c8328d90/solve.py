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
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'Capacity': '180'}}],
             'returned_rows': 1,
             'role': 'storage capacity constraint',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductName', 'Value', 'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
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
             'role': 'bread products and profit coefficients',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    products_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_1_view_0':
            products_table = t
            break
    if products_table is None:
        raise ValueError('Products table not found.')
    capacity_table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            capacity_table = t
            break
    if capacity_table is None:
        raise ValueError('Capacity table not found.')
    P = []
    v_p = {}
    w_p = {}
    for rec in products_table['records']:
        pname = rec['values']['ProductName']
        P.append(pname)
        try:
            v_p[pname] = float(rec['values']['Value'])
            w_p[pname] = float(rec['values']['Weight'])
        except Exception as e:
            raise ValueError(f'Invalid Value or Weight for product {pname}: {e}')
    if len(capacity_table['records']) != 1:
        raise ValueError('Capacity table must have exactly one record.')
    try:
        C = float(capacity_table['records'][0]['values']['Capacity'])
    except Exception as e:
        raise ValueError(f'Invalid Capacity value: {e}')
    for p in P:
        if p not in v_p or p not in w_p:
            raise ValueError(f'Missing profit or weight for product {p}')
    m = gp.Model('bakery_bread_order')
    m.Params.MIPGap = 0.0001
    x = m.addVars(P, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((v_p[p] * x[p] for p in P)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((w_p[p] * x[p] for p in P)) <= C, name='storage_capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for p in P:
            print(x[p].VarName, x[p].X)
    else:
        print(m.Status)
    return m
m = solve_problem()