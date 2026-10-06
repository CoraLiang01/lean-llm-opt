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
import gurobipy as gp
from gurobipy import GRB

def solve_problem(CSVQA_DATA):
    table_id = 'file_0_view_0'
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    if table_id not in tables:
        raise ValueError(f'Table {table_id} not found in CSVQA_DATA')
    records = tables[table_id]['records']
    I = []
    R = {}
    S = {}
    D = {}
    for rec in records:
        vals = rec['values']
        pname = vals['Product_Name']
        if not pname.startswith('Books'):
            continue
        I.append(pname)
        try:
            R[pname] = float(vals['Revenue'])
            S[pname] = float(vals['Initial Inventory'])
            D[pname] = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Error parsing record {pname}: {e}')
    for pname in I:
        if pname not in R or pname not in S or pname not in D:
            raise ValueError(f'Missing data for product {pname}')
    m = gp.Model()
    x = m.addVars(I, lb=0, ub={i: min(int(S[i]), int(D[i])) for i in I}, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((R[i] * x[i] for i in I)), GRB.MAXIMIZE)
    for i in I:
        m.addConstr(x[i] <= S[i], name=f'inv_{i}')
        m.addConstr(x[i] <= D[i], name=f'dem_{i}')
        m.addConstr(x[i] >= 0, name=f'nonneg_{i}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem(CSVQA_DATA)