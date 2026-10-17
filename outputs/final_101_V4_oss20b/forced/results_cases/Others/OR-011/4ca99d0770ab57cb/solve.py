CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the '
          '‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective '
          'is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. '
          'The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. '
          'During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n'
          '\n'
          'Demand for each product during the sales period is assumed to be deterministic and known in advance, with '
          'demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of '
          'units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative '
          'integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the '
          'fulfillment quantities must satisfy both inventory and demand constraints.',
 'relationships': [],
 'route': 'Others',
 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'OnlineRetailSalesDataset.csv',
             'filters': {'conditions': [{'column': 'id_number',
                                         'dtype': 'string',
                                         'evidence': 'id999',
                                         'format': None,
                                         'inclusive': 'both',
                                         'operator': 'prefix',
                                         'value': 'id999'}],
                         'logic': 'and'},
             'original_rows': 900,
             'records': [{'source_row': 899,
                          'values': {'Demand': '8171',
                                     'Initial Inventory': '56450',
                                     'Revenue': '434.74',
                                     'id_number': 'id999'}}],
             'returned_rows': 1,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB
CSVQA_DATA = {'ignored_file_indices': [], 'query': 'The supermarket offers a variety of top-selling products, with associated revenue data provided in the ‘Revenue’ column. Each product has its own demand level during the sales horizon. The company’s objective is to maximize total revenue by allocating the available inventory of products classified under ‘id999’. The initial inventory levels for the ‘id999’ products are detailed in the ‘Initial Inventory’ column. During the sales horizon, no restocking is allowed, and there are no in-transit inventories.\n\nDemand for each product during the sales period is assumed to be deterministic and known in advance, with demand quantities specified in the ‘Demand’ column. The decision variables x_i represent the number of units of each ‘id999’ product i that the company plans to fulfill, where each x_i is a non-negative integer. Because fulfilled orders cannot exceed either the available inventory or the realized demand, the fulfillment quantities must satisfy both inventory and demand constraints.', 'relationships': [], 'route': 'Others', 'tables': [{'columns': ['id_number', 'Revenue', 'Demand', 'Initial Inventory'], 'file_index': 0, 'file_name': 'OnlineRetailSalesDataset.csv', 'filters': {'conditions': [{'column': 'id_number', 'dtype': 'string', 'evidence': 'id999', 'format': None, 'inclusive': 'both', 'operator': 'prefix', 'value': 'id999'}], 'logic': 'and'}, 'original_rows': 900, 'records': [{'source_row': 899, 'values': {'Demand': '8171', 'Initial Inventory': '56450', 'Revenue': '434.74', 'id_number': 'id999'}}], 'returned_rows': 1, 'role': 'descriptive non-unique role', 'table_id': 'file_0_view_0'}], 'validation': {'matrix_checks': [], 'status': 'OK'}}

def solve_problem():
    table = None
    for tbl in CSVQA_DATA.get('tables', []):
        if tbl.get('table_id') == 'file_0_view_0':
            table = tbl
            break
    if table is None:
        raise ValueError("Required table 'file_0_view_0' not found in CSVQA_DATA")
    I = set()
    Rev = {}
    Dem = {}
    Inv = {}
    for record in table.get('records', []):
        values = record.get('values', {})
        id_number = values.get('id_number', '')
        if not isinstance(id_number, str) or not id_number.startswith('id999'):
            continue
        source_row = record.get('source_row')
        if source_row is None:
            raise ValueError('Record missing source_row identifier')
        try:
            rev_val = float(values['Revenue'])
            dem_val = int(values['Demand'])
            inv_val = int(values['Initial Inventory'])
        except KeyError as e:
            raise ValueError(f'Missing required parameter in record {source_row}: {e}')
        except ValueError as e:
            raise ValueError(f'Invalid numeric value in record {source_row}: {e}')
        I.add(source_row)
        Rev[source_row] = rev_val
        Dem[source_row] = dem_val
        Inv[source_row] = inv_val
    if not I:
        raise ValueError("No records satisfy the id_number prefix filter 'id999'")
    m = gurobipy.Model('MaxRevenue')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(I, vtype=GRB.INTEGER, name='')
    for i in I:
        m.addConstr(x[i] <= Inv[i], name=f'inv_{i}')
        m.addConstr(x[i] <= Dem[i], name=f'dem_{i}')
    objective = gurobipy.quicksum((Rev[i] * x[i] for i in I))
    m.setObjective(objective, GRB.MAXIMIZE)
    m.optimize()
    return m
m = solve_problem()