CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with revenue data provided in the ‘Revenue’ '
          'column. Each product has its own demand level. The retailer aims to maximize total revenue by focusing on '
          'the initial inventory of products classified under ‘27in’. Inventory levels are provided in the ‘Initial '
          'Inventory’ column. Demand quantities for ‘27in’ products are given in the ‘Demand’ column and are assumed '
          'to be deterministic and known in advance. The decision variables x_i represent the number of units of each '
          '‘27in’ product i that the company plans to fulfill.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'SalesDataAnalysis.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '27in',
                                         'inclusive': 'both',
                                         'operator': 'contains',
                                         'value': '27in'}],
                         'logic': 'and'},
             'original_rows': 19,
             'records': [{'source_row': 1,
                          'values': {'Demand': '12474',
                                     'Initial Inventory': '62440',
                                     'Product Name': '27in 4K Gaming Monitor',
                                     'Revenue': '389.99'}},
                         {'source_row': 2,
                          'values': {'Demand': '15057',
                                     'Initial Inventory': '75500',
                                     'Product Name': '27in FHD Monitor',
                                     'Revenue': '149.99'}}],
             'returned_rows': 2,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table = None
    for tbl in CSVQA_DATA['tables']:
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    records = table.get('records', [])
    if not records:
        raise ValueError('No records found in the required table')
    I = []
    r = {}
    d = {}
    s = {}
    for rec in records:
        vals = rec.get('values', {})
        prod_name = vals.get('Product Name')
        if prod_name is None:
            raise ValueError(f"Missing 'Product Name' in record {rec.get('source_row')}")
        try:
            revenue = float(vals.get('Revenue', '0'))
            demand = int(vals.get('Demand', '0'))
            inventory = int(vals.get('Initial Inventory', '0'))
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid numeric value in record {rec.get('source_row')}: {e}')
        I.append(prod_name)
        r[prod_name] = revenue
        d[prod_name] = demand
        s[prod_name] = inventory
    for prod in I:
        if prod not in r or prod not in d or prod not in s:
            raise ValueError(f'Missing parameter for product {prod}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(I, vtype=GRB.INTEGER, name='')
    for prod in I:
        model.addConstr(x[prod] <= d[prod], name=f'max_demand_{prod}')
        model.addConstr(x[prod] <= s[prod], name=f'max_inventory_{prod}')
    model.setObjective(gurobipy.quicksum((r[prod] * x[prod] for prod in I)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()