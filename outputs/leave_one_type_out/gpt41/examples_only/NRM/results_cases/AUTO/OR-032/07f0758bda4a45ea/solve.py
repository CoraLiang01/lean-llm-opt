CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers various products with revenue data in the ‘Revenue’ column. The company aims to '
          'maximize total revenue by focusing on products classified under ‘Books’. Inventory levels are detailed in '
          'the ‘Initial Inventory’ column. Demand quantities are specified in the ‘Demand’ column and are assumed to '
          'be deterministic and known in advance. Decision variables x_i represent the number of units of each ‘Books’ '
          'product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'DifferentStoreSales.csv',
             'filters': {'conditions': [{'column': 'Product_Name',
                                         'dtype': 'string',
                                         'evidence': "'Books'",
                                         'operator': 'prefix',
                                         'value': 'Books'}],
                         'logic': 'and'},
             'original_rows': 40,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1980',
                                     'Initial Inventory': '9920.0',
                                     'Product_Name': 'Books_15.15',
                                     'Revenue': '15.15'}},
                         {'source_row': 1,
                          'values': {'Demand': '3024',
                                     'Initial Inventory': '20160.0',
                                     'Product_Name': 'Books_30.3',
                                     'Revenue': '30.3'}},
                         {'source_row': 2,
                          'values': {'Demand': '4536',
                                     'Initial Inventory': '30000.0',
                                     'Product_Name': 'Books_45.45',
                                     'Revenue': '45.45'}},
                         {'source_row': 3,
                          'values': {'Demand': '5601',
                                     'Initial Inventory': '38360.0',
                                     'Product_Name': 'Books_60.6',
                                     'Revenue': '60.6'}},
                         {'source_row': 4,
                          'values': {'Demand': '7567',
                                     'Initial Inventory': '51450.0',
                                     'Product_Name': 'Books_75.75',
                                     'Revenue': '75.75'}}],
             'returned_rows': 5,
             'role': 'product revenue, demand, and inventory',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB

def solve_books_revenue(CSVQA_DATA):
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id not found in CSVQA_DATA.')
    records = table['records']
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    books_prefix = re.compile('^Books')
    for rec in records:
        vals = rec['values']
        pname = vals['Product_Name']
        if not books_prefix.match(pname):
            continue
        try:
            revenue = float(vals['Revenue'])
            initial_inventory = float(vals['Initial Inventory'])
            demand = float(vals['Demand'])
        except Exception as e:
            raise RuntimeError(f'Invalid data for product {pname}: {e}')
        I.append(pname)
        Revenue[pname] = revenue
        InitialInventory[pname] = initial_inventory
        Demand[pname] = demand
    for pname in I:
        if pname not in Revenue or pname not in InitialInventory or pname not in Demand:
            raise RuntimeError(f'Missing data for product {pname}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = {}
    for pname in I:
        ub = min(InitialInventory[pname], Demand[pname])
        x[pname] = m.addVar(vtype=GRB.INTEGER, lb=0, ub=ub, name='x')
    m.update()
    m.setObjective(sum((Revenue[pname] * x[pname] for pname in I)), GRB.MAXIMIZE)
    for pname in I:
        m.addConstr(x[pname] <= InitialInventory[pname], name='inv')
        m.addConstr(x[pname] <= Demand[pname], name='dem')
        m.addConstr(x[pname] >= 0, name='nonneg')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for pname in I:
            print(f'{x[pname].VarName} {x[pname].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_books_revenue(CSVQA_DATA)