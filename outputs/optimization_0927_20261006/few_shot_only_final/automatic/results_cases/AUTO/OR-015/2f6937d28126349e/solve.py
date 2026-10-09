CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'RestaurantSalesreport.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 7,
             'returned_rows': 7,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Aalopuri',
                                     'Revenue': '20',
                                     'Demand': '1483',
                                     'Initial Inventory': '10440.0'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Cold coffee',
                                     'Revenue': '40',
                                     'Demand': '1918',
                                     'Initial Inventory': '13610.0'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Frankie',
                                     'Revenue': '50',
                                     'Demand': '1623',
                                     'Initial Inventory': '11500.0'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Panipuri',
                                     'Revenue': '20',
                                     'Demand': '1720',
                                     'Initial Inventory': '12260.0'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Sandwich',
                                     'Revenue': '60',
                                     'Demand': '1558',
                                     'Initial Inventory': '10970.0'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Sugarcane juice',
                                     'Revenue': '25',
                                     'Demand': '1791',
                                     'Initial Inventory': '12780.0'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Vadapav',
                                     'Revenue': '20',
                                     'Demand': '1426',
                                     'Initial Inventory': '10060.0'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import re
import gurobipy as gp
from gurobipy import GRB

def parse_table_records(table, product_filter=None):
    records = table['records']
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        pname = rec['values']['Product Name']
        if product_filter is not None and product_filter not in pname:
            continue
        items.append(pname)
        try:
            revenue[pname] = float(rec['values']['Revenue'])
        except Exception:
            raise ValueError(f"Revenue missing or invalid for product '{pname}'")
        try:
            demand[pname] = float(rec['values']['Demand'])
        except Exception:
            raise ValueError(f"Demand missing or invalid for product '{pname}'")
        try:
            inventory[pname] = float(rec['values']['Initial Inventory'])
        except Exception:
            raise ValueError(f"Initial Inventory missing or invalid for product '{pname}'")
    return (items, revenue, demand, inventory)

def build_and_solve_model(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Required table_id 'file_0_view_0' not found in CSVQA_DATA")
    (items, revenue, demand, inventory) = parse_table_records(table, product_filter='Aalop')
    if not items:
        raise ValueError("No products found with 'Aalop' in Product Name")
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing data for product '{i}'")
    m = gp.Model('Aalop_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(items, lb=0, ub=[min(demand[i], inventory[i]) for i in items], vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in x_vars.values():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve_model(CSVQA_DATA)