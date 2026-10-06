CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The company operates in the European market and offers a variety of products with revenue data provided in '
          'the ‘Revenue’ column. The company aims to maximize total revenue using the initial inventory of products '
          'classified under ‘Baby’. Inventory levels are provided in the ‘Initial Inventory’ column. Demand quantities '
          'are specified in the ‘Demand’ column and are assumed to be deterministic and known in advance. Decision '
          'variables x_i represent the number of units of each ‘Baby’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'products classified under ‘Baby’',
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'Baby'}],
                         'logic': 'and'},
             'original_rows': 12,
             'records': [{'source_row': 0,
                          'values': {'Demand': '765850',
                                     'Initial Inventory': '5627060',
                                     'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28'}}],
             'returned_rows': 1,
             'role': 'product revenue, demand, and inventory for Baby products',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB

def solve_baby_product_revenue(CSVQA_DATA):
    table_id = 'file_0_view_0'
    tables = {t['table_id']: t for t in CSVQA_DATA['tables']}
    if table_id not in tables:
        raise ValueError(f'Required table_id {table_id} not found in CSVQA_DATA.')
    table = tables[table_id]
    records = table['records']
    I = []
    Revenue = {}
    InitialInventory = {}
    Demand = {}
    for rec in records:
        vals = rec['values']
        pname = vals['Product Name']
        if not pname.startswith('Baby'):
            continue
        I.append(pname)
        try:
            Revenue[pname] = float(vals['Revenue'])
            InitialInventory[pname] = int(vals['Initial Inventory'])
            Demand[pname] = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Error parsing parameters for product {pname}: {e}')
    for pname in I:
        if pname not in Revenue or pname not in InitialInventory or pname not in Demand:
            raise ValueError(f'Missing parameter for product {pname}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, vtype=GRB.INTEGER, lb=0, ub={i: min(InitialInventory[i], Demand[i]) for i in I}, name='')
    m.setObjective(sum((Revenue[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= InitialInventory[i] for i in I), name='')
    m.addConstrs((x[i] <= Demand[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in I:
            print(f'{x[i].VarName}: {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_baby_product_revenue(CSVQA_DATA)