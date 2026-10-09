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
                                         'evidence': 'products classified under ‘Books’',
                                         'inclusive': 'both',
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
             'role': 'product revenue and constraints',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise RuntimeError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    for rec in table['records']:
        v = rec['values']
        prod = v['Product_Name']
        try:
            rev = float(v['Revenue'])
            inv = float(v['Initial Inventory'])
            dem = float(v['Demand'])
        except Exception as e:
            raise RuntimeError(f'Non-numeric data for product {prod}: {e}')
        I.append(prod)
        Revenue[prod] = rev
        InitialInventory[prod] = inv
        Demand[prod] = dem
    if not set(Revenue) == set(InitialInventory) == set(Demand) == set(I):
        raise RuntimeError('Mismatch in parameter keys and index set I.')
    m = Model()
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name='')
    for i in I:
        m.addConstr(x_vars[i] <= InitialInventory[i], name='inv_%s' % i)
        m.addConstr(x_vars[i] <= Demand[i], name='dem_%s' % i)
    m.setObjective(quicksum((Revenue[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal {m.ObjVal}')
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve()