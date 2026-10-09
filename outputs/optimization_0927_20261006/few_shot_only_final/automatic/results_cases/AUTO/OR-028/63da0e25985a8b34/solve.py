CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'WomenClothingEcommerceSalesData.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 24,
             'returned_rows': 24,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'sku_I27',
                                     'Revenue': '238',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'sku_I499',
                                     'Revenue': '287',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'sku_I719',
                                     'Revenue': '268',
                                     'Demand': '16',
                                     'Initial Inventory': '80'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'sku_T18',
                                     'Revenue': '318',
                                     'Demand': '14',
                                     'Initial Inventory': '70'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'sku_T29',
                                     'Revenue': '207',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'sku_T39',
                                     'Revenue': '258',
                                     'Demand': '32',
                                     'Initial Inventory': '160'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'sku_T499',
                                     'Revenue': '249',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'sku_T9',
                                     'Revenue': '227',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'sku_3081',
                                     'Revenue': '198',
                                     'Demand': '10',
                                     'Initial Inventory': '50'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'sku_339',
                                     'Revenue': '254',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'sku_3799',
                                     'Revenue': '246',
                                     'Demand': '18',
                                     'Initial Inventory': '90'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'sku_439',
                                     'Revenue': '258',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 12,
                          'values': {'Product Name': 'sku_539',
                                     'Revenue': '268',
                                     'Demand': '4',
                                     'Initial Inventory': '20'}},
                         {'source_row': 13,
                          'values': {'Product Name': 'sku_61399',
                                     'Revenue': '278',
                                     'Demand': '8',
                                     'Initial Inventory': '40'}},
                         {'source_row': 14,
                          'values': {'Product Name': 'sku_628',
                                     'Revenue': '268',
                                     'Demand': '2',
                                     'Initial Inventory': '10'}},
                         {'source_row': 15,
                          'values': {'Product Name': 'sku_708',
                                     'Revenue': '298',
                                     'Demand': '198',
                                     'Initial Inventory': '990'}},
                         {'source_row': 16,
                          'values': {'Product Name': 'sku_77',
                                     'Revenue': '258',
                                     'Demand': '32',
                                     'Initial Inventory': '160'}},
                         {'source_row': 17,
                          'values': {'Product Name': 'sku_79',
                                     'Revenue': '315',
                                     'Demand': '18',
                                     'Initial Inventory': '90'}},
                         {'source_row': 18,
                          'values': {'Product Name': 'sku_799',
                                     'Revenue': '264',
                                     'Demand': '570',
                                     'Initial Inventory': '2870'}},
                         {'source_row': 19,
                          'values': {'Product Name': 'sku_8499',
                                     'Revenue': '238',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 20,
                          'values': {'Product Name': 'sku_89',
                                     'Revenue': '258',
                                     'Demand': '26',
                                     'Initial Inventory': '130'}},
                         {'source_row': 21,
                          'values': {'Product Name': 'sku_897',
                                     'Revenue': '268',
                                     'Demand': '6',
                                     'Initial Inventory': '30'}},
                         {'source_row': 22,
                          'values': {'Product Name': 'sku_9699',
                                     'Revenue': '288',
                                     'Demand': '33',
                                     'Initial Inventory': '170'}},
                         {'source_row': 23,
                          'values': {'Product Name': 'sku_bobo',
                                     'Revenue': '228',
                                     'Demand': '33',
                                     'Initial Inventory': '170'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import re
import gurobipy as gp
from gurobipy import GRB

def parse_table_records(table, required_columns):
    for col in required_columns:
        if col not in table['columns']:
            raise ValueError(f"Missing required column '{col}' in table '{table['table_id']}'")
    records = []
    for rec in table['records']:
        values = rec['values']
        record = {col: values[col] for col in required_columns}
        records.append(record)
    return records

def get_table_by_id(data, table_id):
    for t in data['tables']:
        if t['table_id'] == table_id:
            return t
    raise ValueError(f"Table with id '{table_id}' not found.")

def build_and_solve_model(CSVQA_DATA):
    table_id = 'file_0_view_0'
    columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    table = get_table_by_id(CSVQA_DATA, table_id)
    records = parse_table_records(table, columns)
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in records:
        prod = rec['Product Name']
        if prod in items:
            raise ValueError(f"Duplicate product name '{prod}' in table '{table_id}'")
        items.append(prod)
        try:
            revenue[prod] = float(rec['Revenue'])
        except Exception:
            raise ValueError(f"Invalid Revenue for product '{prod}': {rec['Revenue']}")
        try:
            demand[prod] = float(rec['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for product '{prod}': {rec['Demand']}")
        try:
            inventory[prod] = float(rec['Initial Inventory'])
        except Exception:
            raise ValueError(f"Invalid Initial Inventory for product '{prod}': {rec['Initial Inventory']}")
    if not set(items) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
        raise ValueError('Mismatch in index sets for products and parameters.')
    m = gp.Model('WomenClothingEcommerceSales')
    x_vars = m.addVars(items, lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve_model(CSVQA_DATA)