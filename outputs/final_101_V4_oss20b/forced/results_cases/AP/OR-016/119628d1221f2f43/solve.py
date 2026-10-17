CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A retail establishment seeks to optimize the allocation of its merchandise across distinct categories '
          '(electronics, apparel, homeware, etc.) to achieve the highest possible total revenue. Each category '
          'maintains its own demand level, with revenue figures provided in the ‘Revenue’ column and current stock '
          'quantities documented in the ‘Initial Inventory’ column. Demand quantities are provided in the ‘Demand’ '
          'column and are assumed to be deterministic and known in advance. The optimization challenge involves '
          'determining the ideal fulfillment quantities x_i for each product that allocate available inventory while '
          'respecting inventory limits and maximizing total revenue.',
 'relationships': [],
 'route': 'AP',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RetailSalesDataset.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 35,
             'records': [{'source_row': 0,
                          'values': {'Demand': '240',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Beauty - 25',
                                     'Revenue': '25'}},
                         {'source_row': 1,
                          'values': {'Demand': '202',
                                     'Initial Inventory': '1330',
                                     'Product Name': 'Beauty - 30',
                                     'Revenue': '30'}},
                         {'source_row': 2,
                          'values': {'Demand': '216',
                                     'Initial Inventory': '1420',
                                     'Product Name': 'Beauty - 300',
                                     'Revenue': '300'}},
                         {'source_row': 3,
                          'values': {'Demand': '263',
                                     'Initial Inventory': '1700',
                                     'Product Name': 'Beauty - 50',
                                     'Revenue': '50'}},
                         {'source_row': 4,
                          'values': {'Demand': '256',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Beauty - 500',
                                     'Revenue': '500'}},
                         {'source_row': 5,
                          'values': {'Demand': '281',
                                     'Initial Inventory': '1840',
                                     'Product Name': 'Clothing - 25',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '261',
                                     'Initial Inventory': '1710',
                                     'Product Name': 'Clothing - 30',
                                     'Revenue': '30'}},
                         {'source_row': 7,
                          'values': {'Demand': '295',
                                     'Initial Inventory': '1930',
                                     'Product Name': 'Clothing - 300',
                                     'Revenue': '300'}},
                         {'source_row': 8,
                          'values': {'Demand': '290',
                                     'Initial Inventory': '1890',
                                     'Product Name': 'Clothing - 50',
                                     'Revenue': '50'}},
                         {'source_row': 9,
                          'values': {'Demand': '244',
                                     'Initial Inventory': '1570',
                                     'Product Name': 'Clothing - 500',
                                     'Revenue': '500'}},
                         {'source_row': 10,
                          'values': {'Demand': '273',
                                     'Initial Inventory': '1810',
                                     'Product Name': 'Electronics - 25',
                                     'Revenue': '25'}},
                         {'source_row': 11,
                          'values': {'Demand': '220',
                                     'Initial Inventory': '1410',
                                     'Product Name': 'Electronics - 30',
                                     'Revenue': '30'}},
                         {'source_row': 12,
                          'values': {'Demand': '286',
                                     'Initial Inventory': '1830',
                                     'Product Name': 'Electronics - 300',
                                     'Revenue': '300'}},
                         {'source_row': 13,
                          'values': {'Demand': '268',
                                     'Initial Inventory': '1750',
                                     'Product Name': 'Electronics - 50',
                                     'Revenue': '50'}},
                         {'source_row': 14,
                          'values': {'Demand': '262',
                                     'Initial Inventory': '1690',
                                     'Product Name': 'Electronics - 500',
                                     'Revenue': '500'}},
                         {'source_row': 15,
                          'values': {'Demand': '255',
                                     'Initial Inventory': '1660',
                                     'Product Name': 'Home Goods - 25',
                                     'Revenue': '25'}},
                         {'source_row': 16,
                          'values': {'Demand': '218',
                                     'Initial Inventory': '1417',
                                     'Product Name': 'Home Goods - 30',
                                     'Revenue': '30'}},
                         {'source_row': 17,
                          'values': {'Demand': '278',
                                     'Initial Inventory': '1807',
                                     'Product Name': 'Home Goods - 50',
                                     'Revenue': '50'}},
                         {'source_row': 18,
                          'values': {'Demand': '195',
                                     'Initial Inventory': '1268',
                                     'Product Name': 'Home Goods - 300',
                                     'Revenue': '300'}},
                         {'source_row': 19,
                          'values': {'Demand': '248',
                                     'Initial Inventory': '1612',
                                     'Product Name': 'Home Goods - 500',
                                     'Revenue': '500'}},
                         {'source_row': 20,
                          'values': {'Demand': '269',
                                     'Initial Inventory': '1749',
                                     'Product Name': 'Sports - 25',
                                     'Revenue': '25'}},
                         {'source_row': 21,
                          'values': {'Demand': '227',
                                     'Initial Inventory': '1476',
                                     'Product Name': 'Sports - 30',
                                     'Revenue': '30'}},
                         {'source_row': 22,
                          'values': {'Demand': '285',
                                     'Initial Inventory': '1853',
                                     'Product Name': 'Sports - 50',
                                     'Revenue': '50'}},
                         {'source_row': 23,
                          'values': {'Demand': '208',
                                     'Initial Inventory': '1352',
                                     'Product Name': 'Sports - 300',
                                     'Revenue': '300'}},
                         {'source_row': 24,
                          'values': {'Demand': '259',
                                     'Initial Inventory': '1684',
                                     'Product Name': 'Sports - 500',
                                     'Revenue': '500'}},
                         {'source_row': 25,
                          'values': {'Demand': '242',
                                     'Initial Inventory': '1573',
                                     'Product Name': 'Furniture - 25',
                                     'Revenue': '25'}},
                         {'source_row': 26,
                          'values': {'Demand': '213',
                                     'Initial Inventory': '1385',
                                     'Product Name': 'Furniture - 30',
                                     'Revenue': '30'}},
                         {'source_row': 27,
                          'values': {'Demand': '272',
                                     'Initial Inventory': '1768',
                                     'Product Name': 'Furniture - 50',
                                     'Revenue': '50'}},
                         {'source_row': 28,
                          'values': {'Demand': '224',
                                     'Initial Inventory': '1456',
                                     'Product Name': 'Furniture - 300',
                                     'Revenue': '300'}},
                         {'source_row': 29,
                          'values': {'Demand': '251',
                                     'Initial Inventory': '1632',
                                     'Product Name': 'Furniture - 500',
                                     'Revenue': '500'}},
                         {'source_row': 30,
                          'values': {'Demand': '257',
                                     'Initial Inventory': '1671',
                                     'Product Name': 'Toys - 25',
                                     'Revenue': '25'}},
                         {'source_row': 31,
                          'values': {'Demand': '235',
                                     'Initial Inventory': '1528',
                                     'Product Name': 'Toys - 30',
                                     'Revenue': '30'}},
                         {'source_row': 32,
                          'values': {'Demand': '291',
                                     'Initial Inventory': '1892',
                                     'Product Name': 'Toys - 50',
                                     'Revenue': '50'}},
                         {'source_row': 33,
                          'values': {'Demand': '199',
                                     'Initial Inventory': '1294',
                                     'Product Name': 'Toys - 300',
                                     'Revenue': '300'}},
                         {'source_row': 34,
                          'values': {'Demand': '264',
                                     'Initial Inventory': '1716',
                                     'Product Name': 'Toys - 500',
                                     'Revenue': '500'}}],
             'returned_rows': 35,
             'role': 'descriptive non-unique role',
             'table_id': 'file_0_view_0'}],
 'validation': {'matrix_checks': [], 'status': 'OK'}}
import gurobipy
from gurobipy import GRB

def solve_problem():
    target_table_id = 'file_0_view_0'
    tables = CSVQA_DATA.get('tables', [])
    target_table = None
    for tbl in tables:
        if tbl.get('table_id') == target_table_id:
            target_table = tbl
            break
    if target_table is None:
        raise ValueError(f'Table with id {target_table_id} not found in CSVQA_DATA')
    records = target_table.get('records', [])
    if not records:
        raise ValueError('No records found in the target table')
    products = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec.get('values', {})
        prod = vals.get('Product Name')
        rev = vals.get('Revenue')
        dem = vals.get('Demand')
        inv = vals.get('Initial Inventory')
        if prod is None or rev is None or dem is None or (inv is None):
            raise ValueError(f'Missing required field in record {rec.get('source_row')}')
        try:
            rev_val = float(rev)
            dem_val = float(dem)
            inv_val = float(inv)
        except ValueError as e:
            raise ValueError(f'Non-numeric value in record {rec.get('source_row')}: {e}')
        if prod in products:
            raise ValueError(f'Duplicate product name detected: {prod}')
        products.append(prod)
        revenue[prod] = rev_val
        demand[prod] = dem_val
        inventory[prod] = inv_val
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    x = model.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
    model.setObjective(gurobipy.quicksum((revenue[p] * x[p] for p in products)), GRB.MAXIMIZE)
    for p in products:
        model.addConstr(x[p] <= demand[p], name='')
        model.addConstr(x[p] <= inventory[p], name='')
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        print(f'ObjVal: {model.ObjVal}')
        for var in model.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {model.Status}')
    return model
m = solve_problem()