CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A restaurant offers a variety of popular products, including fast food and beverages. The profit data for '
          'these products is provided in the ‘Revenue’ column. Each product has its own demand level. The restaurant '
          'aims to maximize total revenue by focusing on the initial inventory of products classified under ‘Aalop’, '
          'which are detailed in the ‘Initial Inventory’ column. During the sales period, restocking is not permitted, '
          'and there are no in-transit inventories. Demand for ‘Aalop’ products during the sales horizon is assumed to '
          'be deterministic and known in advance, with demand information specified in the ‘Demand’ column. The '
          'variables x_i represent the number of units of each ‘Aalop’ product i that the restaurant intends to '
          'fulfill.',
 'relationships': [],
 'route': 'NRM',
 'tables': [{'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 7,
             'records': [{'source_row': 0,
                          'values': {'Demand': '1483',
                                     'Initial Inventory': '10440.0',
                                     'Product Name': 'Aalopuri',
                                     'Revenue': '20'}},
                         {'source_row': 1,
                          'values': {'Demand': '1918',
                                     'Initial Inventory': '13610.0',
                                     'Product Name': 'Cold coffee',
                                     'Revenue': '40'}},
                         {'source_row': 2,
                          'values': {'Demand': '1623',
                                     'Initial Inventory': '11500.0',
                                     'Product Name': 'Frankie',
                                     'Revenue': '50'}},
                         {'source_row': 3,
                          'values': {'Demand': '1720',
                                     'Initial Inventory': '12260.0',
                                     'Product Name': 'Panipuri',
                                     'Revenue': '20'}},
                         {'source_row': 4,
                          'values': {'Demand': '1558',
                                     'Initial Inventory': '10970.0',
                                     'Product Name': 'Sandwich',
                                     'Revenue': '60'}},
                         {'source_row': 5,
                          'values': {'Demand': '1791',
                                     'Initial Inventory': '12780.0',
                                     'Product Name': 'Sugarcane juice',
                                     'Revenue': '25'}},
                         {'source_row': 6,
                          'values': {'Demand': '1426',
                                     'Initial Inventory': '10060.0',
                                     'Product Name': 'Vadapav',
                                     'Revenue': '20'}}],
             'returned_rows': 7,
             'role': 'file_0',
             'table_id': 'file_0_view_0'}],
 'validation': {'status': 'OK'}}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    data = CSVQA_DATA
    table = None
    for t in data['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA.")
    records = table['records']
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        vals = rec['values']
        pname = vals['Product Name']
        try:
            A_i = float(vals['Revenue'])
            d_i = float(vals['Demand'])
            I_i = float(vals['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Invalid data for product '{pname}': {e}")
        items.append(pname)
        revenue[pname] = A_i
        demand[pname] = d_i
        inventory[pname] = I_i
    if not set(items) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
        raise ValueError('Mismatch in index sets for products, revenue, demand, or inventory.')
    m = gp.Model('Aalop_Inventory_Fulfillment')
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()