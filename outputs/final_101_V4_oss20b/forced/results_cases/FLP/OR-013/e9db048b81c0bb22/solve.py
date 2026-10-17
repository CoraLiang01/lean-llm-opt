CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'This supermarket offers a variety of best-selling products, and the specific revenue information is shown '
          'in the table, with relevant data provided in the “Revenue” column. Each product has its own demand level. '
          'The retailer’s objective is to maximize the total revenue by focusing on the sales volume of “4U” products. '
          'The initial inventory levels of these products are detailed in the “Initial Inventory” column. During the '
          'sales period, restocking is not allowed and there is no in-transit inventory.\n'
          '\n'
          'Demand for the “4U” products during the sales horizon is assumed to be deterministic and known in advance, '
          'with demand quantities specified in the “Demand” column. The decision variable x_i represents the number of '
          'units of each “4U” product i that the retailer plans to fulfill, where each x_i is a non-negative integer. '
          'Because the fulfillment quantity cannot exceed either the available inventory or the realized demand, the '
          'decision variables must satisfy both inventory and demand constraints.\n'
          '\n'
          'The retailer therefore aims to determine the optimal fulfillment quantities in order to maximize total '
          'revenue while respecting both inventory availability and demand limits.',
 'relationships': [],
 'route': 'FLP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineSalesinUSA.csv',
             'filters': {'conditions': [{'column': 'Product Name',
                                         'dtype': 'string',
                                         'evidence': '4U',
                                         'operator': 'contains',
                                         'value': '4U'}],
                         'logic': 'and'},
             'original_rows': 47932,
             'records': [{'source_row': 1,
                          'values': {'Demand': '5',
                                     'Initial Inventory': '30',
                                     'Product Name': '4U_Service_22',
                                     'Revenue': '56.0'}},
                         {'source_row': 2,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_36',
                                     'Revenue': '21.6'}},
                         {'source_row': 3,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': '4U_Service_7',
                                     'Revenue': '62.5'}},
                         {'source_row': 9421,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': 'COM4US5A9FCDA605F7B',
                                     'Revenue': '10.0'}},
                         {'source_row': 9422,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': 'COM4US5A9FCF8FE2DE3',
                                     'Revenue': '10.0'}},
                         {'source_row': 11027,
                          'values': {'Demand': '15',
                                     'Initial Inventory': '100',
                                     'Product Name': 'ENT4US5A979B09A2BF7',
                                     'Revenue': '30.0'}},
                         {'source_row': 18013,
                          'values': {'Demand': '11',
                                     'Initial Inventory': '70',
                                     'Product Name': 'KAB4US59AD6CF40C781',
                                     'Revenue': '12.0'}},
                         {'source_row': 18014,
                          'values': {'Demand': '36',
                                     'Initial Inventory': '240',
                                     'Product Name': 'KAB4US59AD6CF9E69D8',
                                     'Revenue': '11.9'}},
                         {'source_row': 18015,
                          'values': {'Demand': '5',
                                     'Initial Inventory': '30',
                                     'Product Name': 'KAB4US59AD6D003A730',
                                     'Revenue': '10.0'}},
                         {'source_row': 18016,
                          'values': {'Demand': '19',
                                     'Initial Inventory': '130',
                                     'Product Name': 'KAB4US59AD6D0489179',
                                     'Revenue': '15.0'}},
                         {'source_row': 18017,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '40',
                                     'Product Name': 'KAB4US59AD6D0B32674',
                                     'Revenue': '14.9'}},
                         {'source_row': 20681,
                          'values': {'Demand': '110',
                                     'Initial Inventory': '710',
                                     'Product Name': 'MAT4US5A8BEC8AB2742',
                                     'Revenue': '45.0'}},
                         {'source_row': 20682,
                          'values': {'Demand': '6',
                                     'Initial Inventory': '40',
                                     'Product Name': 'MAT4US5A9FAC1A50375',
                                     'Revenue': '12.0'}},
                         {'source_row': 20683,
                          'values': {'Demand': '20',
                                     'Initial Inventory': '130',
                                     'Product Name': 'MAT4US5A9FB58986D2E',
                                     'Revenue': '25.0'}},
                         {'source_row': 20684,
                          'values': {'Demand': '68',
                                     'Initial Inventory': '480',
                                     'Product Name': 'MAT4US5A9FBC6918099',
                                     'Revenue': '5.0'}},
                         {'source_row': 20685,
                          'values': {'Demand': '9',
                                     'Initial Inventory': '60',
                                     'Product Name': 'MAT4US5A9FBDFFF21B5',
                                     'Revenue': '7.0'}},
                         {'source_row': 20686,
                          'values': {'Demand': '3',
                                     'Initial Inventory': '20',
                                     'Product Name': 'MAT4US5AABB7579D03C',
                                     'Revenue': '80.0'}},
                         {'source_row': 37437,
                          'values': {'Demand': '19',
                                     'Initial Inventory': '130',
                                     'Product Name': 'WOF4US5A979D79EEF31',
                                     'Revenue': '15.0'}}],
             'returned_rows': 18,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f"Table with id '{table_id}' not found in CSVQA_DATA")
    records = target_table.get('records', [])
    if not records:
        raise ValueError(f"No records found in table '{table_id}'")
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        values = rec.get('values', {})
        prod = values.get('Product Name')
        rev = values.get('Revenue')
        dem = values.get('Demand')
        inv = values.get('Initial Inventory')
        if prod is None or rev is None or dem is None or (inv is None):
            raise ValueError(f'Missing required field in record {rec.get('source_row')}')
        try:
            rev_val = float(rev)
            dem_val = int(dem)
            inv_val = int(inv)
        except ValueError as e:
            raise ValueError(f'Invalid numeric conversion in record {rec.get('source_row')}: {e}')
        products.append(prod)
        revenue[prod] = rev_val
        demand[prod] = dem_val
        inventory[prod] = inv_val
    m = gurobipy.Model('MaxRevenue_4U')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
    for prod in products:
        m.addConstr(x[prod] <= demand[prod], name=f'demand_{prod}')
        m.addConstr(x[prod] <= inventory[prod], name=f'inventory_{prod}')
    m.setObjective(gurobipy.quicksum((revenue[prod] * x[prod] for prod in products)), GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()