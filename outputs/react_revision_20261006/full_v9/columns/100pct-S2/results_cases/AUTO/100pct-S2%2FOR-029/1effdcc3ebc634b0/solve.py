CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'In retail, shops need to allocate various types of products to different displays. The capacity limit of '
          'each display is provided in "capacity.csv", and the value and weight of each product are provided in '
          '"products.csv". The objective is to determine the optimal number of each product to place on each display '
          'so as to maximize the total value of all products placed across the displays, while ensuring that the total '
          'weight of the products on each display does not exceed its capacity. In addition, the total quantity of the '
          'first product placed across all displays must be at least 5. The decision variable x_{ij} represents the '
          'number of units of product j placed on display i.',
 'relationships': [],
 'route': 'RA',
 'tables': [{'columns': ['ShelfID', 'CleaningMinutesLastMonth', 'Capacity', 'CleaningVisitsLastQuarter'],
             'file_index': 0,
             'file_name': 'capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 10,
             'records': [{'source_row': 0,
                          'values': {'Capacity': '5',
                                     'CleaningMinutesLastMonth': '12',
                                     'CleaningVisitsLastQuarter': '22',
                                     'ShelfID': '1'}},
                         {'source_row': 1,
                          'values': {'Capacity': '7',
                                     'CleaningMinutesLastMonth': '25',
                                     'CleaningVisitsLastQuarter': '12',
                                     'ShelfID': '2'}},
                         {'source_row': 2,
                          'values': {'Capacity': '6',
                                     'CleaningMinutesLastMonth': '3',
                                     'CleaningVisitsLastQuarter': '27',
                                     'ShelfID': '3'}},
                         {'source_row': 3,
                          'values': {'Capacity': '8',
                                     'CleaningMinutesLastMonth': '24',
                                     'CleaningVisitsLastQuarter': '22',
                                     'ShelfID': '4'}},
                         {'source_row': 4,
                          'values': {'Capacity': '5.5',
                                     'CleaningMinutesLastMonth': '3',
                                     'CleaningVisitsLastQuarter': '20',
                                     'ShelfID': '5'}},
                         {'source_row': 5,
                          'values': {'Capacity': '9',
                                     'CleaningMinutesLastMonth': '4',
                                     'CleaningVisitsLastQuarter': '24',
                                     'ShelfID': '6'}},
                         {'source_row': 6,
                          'values': {'Capacity': '6.5',
                                     'CleaningMinutesLastMonth': '11',
                                     'CleaningVisitsLastQuarter': '23',
                                     'ShelfID': '7'}},
                         {'source_row': 7,
                          'values': {'Capacity': '7.5',
                                     'CleaningMinutesLastMonth': '6',
                                     'CleaningVisitsLastQuarter': '30',
                                     'ShelfID': '8'}},
                         {'source_row': 8,
                          'values': {'Capacity': '8.2',
                                     'CleaningMinutesLastMonth': '14',
                                     'CleaningVisitsLastQuarter': '11',
                                     'ShelfID': '9'}},
                         {'source_row': 9,
                          'values': {'Capacity': '5.7',
                                     'CleaningMinutesLastMonth': '29',
                                     'CleaningVisitsLastQuarter': '23',
                                     'ShelfID': '10'}}],
             'returned_rows': 10,
             'role': 'file_0',
             'table_id': 'file_0_view_0'},
            {'columns': ['ProductPhotoCount',
                         'CatalogViewsLastMonth',
                         'MerchandisingTeam',
                         'ProductName',
                         'Value',
                         'Weight'],
             'file_index': 1,
             'file_name': 'products.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 20,
             'records': [{'source_row': 0,
                          'values': {'CatalogViewsLastMonth': '14',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Smartphone',
                                     'ProductPhotoCount': '30',
                                     'Value': '200',
                                     'Weight': '1'}},
                         {'source_row': 1,
                          'values': {'CatalogViewsLastMonth': '12',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Laptop',
                                     'ProductPhotoCount': '18',
                                     'Value': '1500',
                                     'Weight': '5'}},
                         {'source_row': 2,
                          'values': {'CatalogViewsLastMonth': '8',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Headphones',
                                     'ProductPhotoCount': '27',
                                     'Value': '100',
                                     'Weight': '0.5'}},
                         {'source_row': 3,
                          'values': {'CatalogViewsLastMonth': '29',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Camera',
                                     'ProductPhotoCount': '2',
                                     'Value': '800',
                                     'Weight': '2'}},
                         {'source_row': 4,
                          'values': {'CatalogViewsLastMonth': '26',
                                     'MerchandisingTeam': 'Team_C',
                                     'ProductName': 'Smartwatch',
                                     'ProductPhotoCount': '16',
                                     'Value': '250',
                                     'Weight': '0.3'}},
                         {'source_row': 5,
                          'values': {'CatalogViewsLastMonth': '18',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Tablet',
                                     'ProductPhotoCount': '16',
                                     'Value': '600',
                                     'Weight': '1.5'}},
                         {'source_row': 6,
                          'values': {'CatalogViewsLastMonth': '7',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Bluetooth Speaker',
                                     'ProductPhotoCount': '28',
                                     'Value': '150',
                                     'Weight': '1'}},
                         {'source_row': 7,
                          'values': {'CatalogViewsLastMonth': '1',
                                     'MerchandisingTeam': 'Team_C',
                                     'ProductName': 'Keyboard',
                                     'ProductPhotoCount': '12',
                                     'Value': '80',
                                     'Weight': '0.8'}},
                         {'source_row': 8,
                          'values': {'CatalogViewsLastMonth': '20',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Mouse',
                                     'ProductPhotoCount': '2',
                                     'Value': '50',
                                     'Weight': '0.2'}},
                         {'source_row': 9,
                          'values': {'CatalogViewsLastMonth': '9',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Monitor',
                                     'ProductPhotoCount': '16',
                                     'Value': '300',
                                     'Weight': '3'}},
                         {'source_row': 10,
                          'values': {'CatalogViewsLastMonth': '17',
                                     'MerchandisingTeam': 'Team_C',
                                     'ProductName': 'Printer',
                                     'ProductPhotoCount': '2',
                                     'Value': '400',
                                     'Weight': '4'}},
                         {'source_row': 11,
                          'values': {'CatalogViewsLastMonth': '7',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'External Hard Drive',
                                     'ProductPhotoCount': '16',
                                     'Value': '120',
                                     'Weight': '0.5'}},
                         {'source_row': 12,
                          'values': {'CatalogViewsLastMonth': '22',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Router',
                                     'ProductPhotoCount': '27',
                                     'Value': '60',
                                     'Weight': '0.3'}},
                         {'source_row': 13,
                          'values': {'CatalogViewsLastMonth': '12',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Power Bank',
                                     'ProductPhotoCount': '12',
                                     'Value': '40',
                                     'Weight': '0.4'}},
                         {'source_row': 14,
                          'values': {'CatalogViewsLastMonth': '11',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Memory Card',
                                     'ProductPhotoCount': '9',
                                     'Value': '30',
                                     'Weight': '0.05'}},
                         {'source_row': 15,
                          'values': {'CatalogViewsLastMonth': '21',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'USB Flash Drive',
                                     'ProductPhotoCount': '26',
                                     'Value': '25',
                                     'Weight': '0.02'}},
                         {'source_row': 16,
                          'values': {'CatalogViewsLastMonth': '13',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Smart Home Hub',
                                     'ProductPhotoCount': '30',
                                     'Value': '100',
                                     'Weight': '0.6'}},
                         {'source_row': 17,
                          'values': {'CatalogViewsLastMonth': '24',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'Gaming Console',
                                     'ProductPhotoCount': '5',
                                     'Value': '500',
                                     'Weight': '4'}},
                         {'source_row': 18,
                          'values': {'CatalogViewsLastMonth': '15',
                                     'MerchandisingTeam': 'Team_B',
                                     'ProductName': 'Fitness Tracker',
                                     'ProductPhotoCount': '9',
                                     'Value': '90',
                                     'Weight': '0.2'}},
                         {'source_row': 19,
                          'values': {'CatalogViewsLastMonth': '28',
                                     'MerchandisingTeam': 'Team_A',
                                     'ProductName': 'E-Reader',
                                     'ProductPhotoCount': '18',
                                     'Value': '180',
                                     'Weight': '0.5'}}],
             'returned_rows': 20,
             'role': 'file_1',
             'table_id': 'file_1_view_0'}],
 'validation': {'fallback_reason': "Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}",
                'planner_errors': ["Relationship references an unknown table_id: {'type': 'matrix', 'matrix_table_id': "
                                   "'file_2_view_0', 'row_id_column': 'ShelfID', 'row_axis': {'table_id': "
                                   "'file_0_view_0', 'id_column': 'ShelfID'}, 'column_axis': {'table_id': "
                                   "'file_1_view_0', 'id_column': 'ProductName'}}"],
                'status': 'FALLBACK_FULL_DATA'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    capacity_frame = CSVQA_FRAMES['file_0_view_0']
    products_frame = CSVQA_FRAMES['file_1_view_0']
    displays = []
    display_capacity = {}
    for (source_row, row) in capacity_frame.iterrows():
        shelf_id = row['ShelfID']
        displays.append(shelf_id)
        try:
            display_capacity[shelf_id] = float(row['Capacity'])
        except Exception:
            raise ValueError(f"Invalid Capacity for ShelfID {shelf_id}: {row['Capacity']}")
    products = []
    product_value = {}
    product_weight = {}
    for (source_row, row) in products_frame.iterrows():
        product_name = row['ProductName']
        products.append(product_name)
        try:
            product_value[product_name] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for ProductName {product_name}: {row['Value']}")
        try:
            product_weight[product_name] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for ProductName {product_name}: {row['Weight']}")
    first_product_row = products_frame.loc[products_frame.index[0]]
    j_star = first_product_row['ProductName']
    if set(displays) != set(display_capacity.keys()):
        raise ValueError('Mismatch in display identifiers and capacities.')
    if set(products) != set(product_value.keys()) or set(products) != set(product_weight.keys()):
        raise ValueError('Mismatch in product identifiers and value/weight.')
    m = gp.Model('retail_display_allocation')
    m.setParam('MIPGap', 0.0001)
    quantity_vars = m.addVars(displays, products, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((product_value[j] * quantity_vars[i, j] for i in displays for j in products)), GRB.MAXIMIZE)
    for i in displays:
        m.addConstr(gp.quicksum((product_weight[j] * quantity_vars[i, j] for j in products)) <= display_capacity[i])
    m.addConstr(gp.quicksum((quantity_vars[i, j_star] for i in displays)) >= 5)
    m.optimize()
    return m
m = solve_problem()