CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A small bakery in South Korea, and each day need to stock up on various types of bread. For each type of '
          'bread, we have an expected profit, which can be found in "products.csv." However, the shop has limited '
          'storage capacity, with details provided in "capacity.csv.".Therefore, we must decide which types of bread '
          'to order each day to maximize our total expected profit while staying within our storage limits. The '
          'decision variables x_i represents the number of units of bread type i to be ordered each day.The decision '
          'variables must be integers.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['resource_capacity'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {},
             'original_rows': 1,
             'records': [{'source_row': 0, 'values': {'resource_capacity': '180'}}],
             'returned_rows': 1,
             'role': 'storage capacity',
             'table_id': 'file_0_view_0'},
            {'columns': ['item_name', 'item_value', 'resource_requirement'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'item_name': 'Baguette', 'item_value': '888', 'resource_requirement': '4'}},
                         {'source_row': 1,
                          'values': {'item_name': 'Croissant', 'item_value': '134', 'resource_requirement': '2'}},
                         {'source_row': 2,
                          'values': {'item_name': 'Sourdough', 'item_value': '129', 'resource_requirement': '4'}},
                         {'source_row': 3,
                          'values': {'item_name': 'Rye Bread', 'item_value': '370', 'resource_requirement': '3'}},
                         {'source_row': 4,
                          'values': {'item_name': 'Brioche', 'item_value': '921', 'resource_requirement': '2'}},
                         {'source_row': 5,
                          'values': {'item_name': 'Focaccia', 'item_value': '765', 'resource_requirement': '1'}},
                         {'source_row': 6,
                          'values': {'item_name': 'Ciabatta', 'item_value': '154', 'resource_requirement': '2'}},
                         {'source_row': 7,
                          'values': {'item_name': 'Pita', 'item_value': '837', 'resource_requirement': '1'}},
                         {'source_row': 8,
                          'values': {'item_name': 'Bagel', 'item_value': '584', 'resource_requirement': '3'}},
                         {'source_row': 9,
                          'values': {'item_name': 'English Muffin', 'item_value': '365', 'resource_requirement': '3'}}],
             'returned_rows': 10,
             'role': 'bread products',
             'table_id': 'file_1_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_FRAMES):
    import pandas as pd
    products_df = CSVQA_FRAMES['file_1_view_0']
    capacity_df = CSVQA_FRAMES['file_0_view_0']
    items = products_df['item_name'].tolist()
    profit = dict(zip(products_df['item_name'], products_df['item_value'].astype(int)))
    space = dict(zip(products_df['item_name'], products_df['resource_requirement'].astype(int)))
    if capacity_df.shape[0] != 1:
        raise ValueError('Expected exactly one row in capacity.csv')
    C = int(capacity_df.iloc[0]['resource_capacity'])
    if set(items) != set(profit.keys()) or set(items) != set(space.keys()):
        raise ValueError('Mismatch in items/profit/space keys')
    m = gp.Model('Bakery_Bread_Order')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((profit[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((space[i] * quantity_vars[i] for i in items)) <= C, name='storage_capacity')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_FRAMES)