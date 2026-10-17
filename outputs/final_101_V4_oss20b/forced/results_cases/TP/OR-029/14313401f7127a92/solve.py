CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'An e-commerce platform sells women’s clothing items with revenue data provided in the ‘Revenue’ column. The '
          'company aims to maximize total revenue using the initial inventory of products classified under ‘FAUX’. '
          'Inventory levels are detailed in the ‘Initial Inventory’ column. Demand quantities are specified in the '
          '‘Demand’ column and are assumed to be deterministic and known in advance. Decision variables x_i represent '
          'the number of units of each ‘FAUX’ product i that will be fulfilled.',
 'relationships': [],
 'route': 'TP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'ZARASales.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': 'FAUX',
                                         'operator': 'contains',
                                         'value': 'FAUX'}],
                         'logic': 'and'},
             'original_rows': 194,
             'records': [{'source_row': 56,
                          'values': {'Demand': '3025',
                                     'Initial Inventory': '20970',
                                     'Product Name': 'FAUX FUR JEWEL SWEATER',
                                     'Revenue': '35.9'}},
                         {'source_row': 57,
                          'values': {'Demand': '9585',
                                     'Initial Inventory': '71970',
                                     'Product Name': 'FAUX LEATHER BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 58,
                          'values': {'Demand': '4486',
                                     'Initial Inventory': '32730',
                                     'Product Name': 'FAUX LEATHER BOXY FIT JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 59,
                          'values': {'Demand': '10322',
                                     'Initial Inventory': '71130',
                                     'Product Name': 'FAUX LEATHER JACKET',
                                     'Revenue': '99.9'}},
                         {'source_row': 60,
                          'values': {'Demand': '4868',
                                     'Initial Inventory': '34910',
                                     'Product Name': 'FAUX LEATHER OVERSIZED JACKET LIMITED EDITION',
                                     'Revenue': '159.0'}},
                         {'source_row': 61,
                          'values': {'Demand': '8482',
                                     'Initial Inventory': '64010',
                                     'Product Name': 'FAUX LEATHER PUFFER JACKET',
                                     'Revenue': '69.99'}},
                         {'source_row': 62,
                          'values': {'Demand': '2607',
                                     'Initial Inventory': '20760',
                                     'Product Name': 'FAUX SHEARLING LINED SUEDE BOOTS',
                                     'Revenue': '99.9'}},
                         {'source_row': 63,
                          'values': {'Demand': '1784',
                                     'Initial Inventory': '12490',
                                     'Product Name': 'FAUX SHEARLING PLAID JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 64,
                          'values': {'Demand': '6626',
                                     'Initial Inventory': '50300',
                                     'Product Name': 'FAUX SUEDE BOMBER JACKET',
                                     'Revenue': '69.9'}},
                         {'source_row': 65,
                          'values': {'Demand': '3256',
                                     'Initial Inventory': '24570',
                                     'Product Name': 'FAUX SUEDE JACKET',
                                     'Revenue': '89.9'}},
                         {'source_row': 66,
                          'values': {'Demand': '2955',
                                     'Initial Inventory': '24430',
                                     'Product Name': 'FAUX SUEDE OVERSHIRT',
                                     'Revenue': '69.9'}},
                         {'source_row': 67,
                          'values': {'Demand': '910',
                                     'Initial Inventory': '7070',
                                     'Product Name': 'FAUX SUEDE PATCH JACKET',
                                     'Revenue': '89.9'}}],
             'returned_rows': 12,
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
    revenue = {}
    demand = {}
    inventory = {}
    product_names = set()
    for rec in records:
        vals = rec.get('values', {})
        for field in ('Product Name', 'Revenue', 'Demand', 'Initial Inventory'):
            if field not in vals:
                raise ValueError(f"Missing field '{field}' in record {rec.get('source_row')}")
        prod = vals['Product Name']
        if prod in product_names:
            raise ValueError(f'Duplicate product name detected: {prod}')
        product_names.add(prod)
        try:
            rev = float(vals['Revenue'])
            dem = float(vals['Demand'])
            inv = float(vals['Initial Inventory'])
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {rec.get('source_row')}: {e}')
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    if set(revenue.keys()) != product_names:
        raise ValueError('Revenue keys mismatch')
    if set(demand.keys()) != product_names:
        raise ValueError('Demand keys mismatch')
    if set(inventory.keys()) != product_names:
        raise ValueError('Inventory keys mismatch')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(product_names, lb=0, vtype=GRB.CONTINUOUS, name='')
    for prod in product_names:
        model.addConstr(x[prod] <= demand[prod], name='')
        model.addConstr(x[prod] <= inventory[prod], name='')
    model.setObjective(gurobipy.quicksum((revenue[prod] * x[prod] for prod in product_names)), GRB.MAXIMIZE)
    model.optimize()
    return model
m = solve_problem()