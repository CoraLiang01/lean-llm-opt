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
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
from gurobipy import Model, GRB, quicksum

def build_and_solve():
    table = CSVQA_DATA['tables'][0]
    records = table['records']
    I = []
    r = {}
    s = {}
    d = {}
    for rec in records:
        vals = rec['values']
        prod = vals['Product Name']
        I.append(prod)
        try:
            r[prod] = float(vals['Revenue'])
            s[prod] = int(vals['Initial Inventory'])
            d[prod] = int(vals['Demand'])
        except Exception as e:
            raise ValueError(f'Invalid data for product {prod}: {e}')
    for prod in I:
        if prod not in r or prod not in s or prod not in d:
            raise ValueError(f'Missing data for product {prod}')
    m = Model()
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(I, vtype=GRB.INTEGER, lb=0, name='')
    for i in I:
        m.addConstr(x_vars[i] <= s[i], name='')
        m.addConstr(x_vars[i] <= d[i], name='')
    m.setObjective(quicksum((r[i] * x_vars[i] for i in I)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(m.ObjVal)
        for i in I:
            print(f'{x_vars[i].VarName} {x_vars[i].X}')
    else:
        print(m.Status)
    return m
m = build_and_solve()