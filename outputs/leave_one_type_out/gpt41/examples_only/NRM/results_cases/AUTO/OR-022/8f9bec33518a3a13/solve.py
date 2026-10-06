CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The department store is hosting a promotional event featuring various top-selling items. Revenue data is '
          'available in the ‘Revenue’ column. The retailer aims to maximize total revenue using the initial inventory '
          'of products classified under ‘27in’. Inventory levels are detailed in the ‘Initial Inventory’ column. '
          'Demand quantities are provided in the ‘Demand’ column and are assumed to be deterministic and known in '
          'advance. Decision variables x_i indicate the number of units of each ‘27in’ product i that will be '
          'fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'Salesorders.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': "'27in' product i",
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '261.2933'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '52.4965'}}],
             'returned_rows': 2,
             'role': 'revenue, demand, and inventory for 27in products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import re
from gurobipy import Model, GRB

def solve_27in_revenue_optimization(CSVQA_DATA):
    table_id = 'file_0_view_0'
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == table_id:
            table = t
            break
    if table is None:
        raise RuntimeError('Required table_id not found in CSVQA_DATA.')
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    prefix_regex = re.compile('^27in')
    for rec in table['records']:
        pname = rec['values']['Product Name']
        if not prefix_regex.match(pname):
            continue
        try:
            revenue = float(rec['values']['Revenue'])
            initial_inventory = int(rec['values']['Initial Inventory'])
            demand = int(rec['values']['Demand'])
        except Exception as e:
            raise RuntimeError(f"Invalid data for product '{pname}': {e}")
        I.append(pname)
        Revenue[pname] = revenue
        InitialInventory[pname] = initial_inventory
        Demand[pname] = demand
    for pname in I:
        if pname not in Revenue or pname not in InitialInventory or pname not in Demand:
            raise RuntimeError(f"Missing parameter for product '{pname}'.")
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
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for pname in I:
            print(f'{x[pname].VarName}: {x[pname].X}')
    else:
        print(f'Solver status: {m.Status}')
m = solve_27in_revenue_optimization(CSVQA_DATA)