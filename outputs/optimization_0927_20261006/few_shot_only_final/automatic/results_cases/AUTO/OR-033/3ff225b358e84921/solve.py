CSVQA_DATA = {'route': 'NRM',
 'tables': [{'table_id': 'file_0_view_0',
             'file_index': 0,
             'file_name': 'EuropeSalesRecords.csv',
             'source': '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv',
             'role': 'file_0',
             'columns': ['Product Name', 'Revenue', 'Demand', 'Initial Inventory'],
             'original_rows': 12,
             'returned_rows': 12,
             'filters': {'logic': 'and', 'conditions': []},
             'records': [{'source_row': 0,
                          'values': {'Product Name': 'Baby Food_255.28',
                                     'Revenue': '255.28',
                                     'Demand': '765850',
                                     'Initial Inventory': '5627060'}},
                         {'source_row': 1,
                          'values': {'Product Name': 'Beverages_47.45',
                                     'Revenue': '47.45',
                                     'Demand': '825453',
                                     'Initial Inventory': '6131330'}},
                         {'source_row': 2,
                          'values': {'Product Name': 'Cereal_205.7',
                                     'Revenue': '205.7',
                                     'Demand': '627481',
                                     'Initial Inventory': '4656850'}},
                         {'source_row': 3,
                          'values': {'Product Name': 'Clothes_109.28',
                                     'Revenue': '109.28',
                                     'Demand': '800987',
                                     'Initial Inventory': '5913850'}},
                         {'source_row': 4,
                          'values': {'Product Name': 'Cosmetics_437.2',
                                     'Revenue': '437.2',
                                     'Demand': '718806',
                                     'Initial Inventory': '5332910'}},
                         {'source_row': 5,
                          'values': {'Product Name': 'Fruits_9.33',
                                     'Revenue': '9.33',
                                     'Demand': '798999',
                                     'Initial Inventory': '5916720'}},
                         {'source_row': 6,
                          'values': {'Product Name': 'Household_668.27',
                                     'Revenue': '668.27',
                                     'Demand': '591313',
                                     'Initial Inventory': '4402490'}},
                         {'source_row': 7,
                          'values': {'Product Name': 'Meat_421.89',
                                     'Revenue': '421.89',
                                     'Demand': '713606',
                                     'Initial Inventory': '5333760'}},
                         {'source_row': 8,
                          'values': {'Product Name': 'Office Supplies_651.21',
                                     'Revenue': '651.21',
                                     'Demand': '838862',
                                     'Initial Inventory': '6176410'}},
                         {'source_row': 9,
                          'values': {'Product Name': 'Personal Care_81.73',
                                     'Revenue': '81.73',
                                     'Demand': '756433',
                                     'Initial Inventory': '5604800'}},
                         {'source_row': 10,
                          'values': {'Product Name': 'Snacks_152.58',
                                     'Revenue': '152.58',
                                     'Demand': '655310',
                                     'Initial Inventory': '4901600'}},
                         {'source_row': 11,
                          'values': {'Product Name': 'Vegetables_154.06',
                                     'Revenue': '154.06',
                                     'Demand': '786187',
                                     'Initial Inventory': '5825440'}}]}],
 'relationships': [],
 'ignored_file_indices': [],
 'validation': {'status': 'PYTHON_FULL_CSV'}}
import gurobipy as gp
from gurobipy import GRB
import re

def parse_baby_products_from_csvqa(CSVQA_DATA):
    table = None
    for t in CSVQA_DATA['tables']:
        if t['table_id'] == 'file_0_view_0':
            table = t
            break
    if table is None:
        raise ValueError("Table with table_id 'file_0_view_0' not found in CSVQA_DATA.")
    columns = table['columns']
    try:
        idx_product = columns.index('Product Name')
        idx_revenue = columns.index('Revenue')
        idx_demand = columns.index('Demand')
        idx_inventory = columns.index('Initial Inventory')
    except ValueError as e:
        raise ValueError(f'Missing required column: {e}')
    baby_regex = re.compile('\\bBaby\\b', re.IGNORECASE)
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for rec in table['records']:
        values = rec['values']
        product_name = values['Product Name']
        if not isinstance(product_name, str):
            continue
        if not baby_regex.search(product_name):
            continue
        try:
            revenue_val = float(values['Revenue'])
            demand_val = int(float(values['Demand']))
            inventory_val = int(float(values['Initial Inventory']))
        except Exception as e:
            raise ValueError(f"Failed to parse numeric fields for product '{product_name}': {e}")
        items.append(product_name)
        revenue[product_name] = revenue_val
        demand[product_name] = demand_val
        inventory[product_name] = inventory_val
    if not items:
        raise ValueError("No products classified as 'Baby' found in the data.")
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f"Missing data for product '{i}'.")
    return (items, revenue, demand, inventory)

def build_and_solve_baby_inventory_model(CSVQA_DATA):
    (items, revenue, demand, inventory) = parse_baby_products_from_csvqa(CSVQA_DATA)
    if not set(items) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
        raise ValueError('Mismatch in index sets for items, revenue, demand, or inventory.')
    m = gp.Model('Baby_Product_Inventory')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in x_vars.values():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = build_and_solve_baby_inventory_model(CSVQA_DATA)