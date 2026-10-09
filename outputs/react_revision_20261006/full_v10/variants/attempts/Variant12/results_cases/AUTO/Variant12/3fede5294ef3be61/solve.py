CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'A snack distributor ships cartons from plants to regional retailers. Plant capacities are listed in '
          'plant_capacity.csv, retailer demands are listed in retailer_demand.csv, variable per-carton route costs are '
          'listed in route_variable_costs.csv, and fixed route activation costs are listed in route_fixed_costs.csv. A '
          'route can carry flow only if it is activated.\n'
          '\n'
          'Formulate a minimum-cost fixed-charge transportation model. For each plant-retailer route i-j, define x_ij '
          'as the nonnegative shipment quantity and y_ij as a binary variable equal to 1 if route i-j is opened. The '
          'objective is to minimize variable transportation cost plus fixed route-opening cost. The model should '
          'include retailer demand constraints, plant supply upper-bound constraints, shipment-to-route activation '
          'linking constraints using M_ij = min(plant capacity_i, retailer demand_j), nonnegativity constraints for '
          'shipment variables, and binary restrictions for route variables.',
 'relationships': [{'column_axis': {'id_column': 'Retailer', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Plant', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Plant',
                    'type': 'matrix'},
                   {'column_axis': {'id_column': 'Retailer', 'table_id': 'file_1_view_0'},
                    'matrix_table_id': 'file_3_view_0',
                    'row_axis': {'id_column': 'Plant', 'table_id': 'file_0_view_0'},
                    'row_id_column': 'Plant',
                    'type': 'matrix'}],
 'route': 'Others',
 'tables': [{'columns': ['Plant', 'SupplyCapacity'],
             'file_index': 0,
             'file_name': 'plant_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0, 'values': {'Plant': 'P1', 'SupplyCapacity': '190'}},
                         {'source_row': 1, 'values': {'Plant': 'P2', 'SupplyCapacity': '160'}},
                         {'source_row': 2, 'values': {'Plant': 'P3', 'SupplyCapacity': '150'}}],
             'returned_rows': 3,
             'role': 'plant capacities',
             'table_id': 'file_0_view_0'},
            {'columns': ['Retailer', 'Demand'],
             'file_index': 1,
             'file_name': 'retailer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 6,
             'records': [{'source_row': 0, 'values': {'Demand': '70', 'Retailer': 'R1'}},
                         {'source_row': 1, 'values': {'Demand': '85', 'Retailer': 'R2'}},
                         {'source_row': 2, 'values': {'Demand': '75', 'Retailer': 'R3'}},
                         {'source_row': 3, 'values': {'Demand': '65', 'Retailer': 'R4'}},
                         {'source_row': 4, 'values': {'Demand': '95', 'Retailer': 'R5'}},
                         {'source_row': 5, 'values': {'Demand': '80', 'Retailer': 'R6'}}],
             'returned_rows': 6,
             'role': 'retailer demands',
             'table_id': 'file_1_view_0'},
            {'columns': ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
             'file_index': 2,
             'file_name': 'route_variable_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Plant': 'P1',
                                     'R1': '3',
                                     'R2': '4',
                                     'R3': '18',
                                     'R4': '20',
                                     'R5': '22',
                                     'R6': '19'}},
                         {'source_row': 1,
                          'values': {'Plant': 'P2',
                                     'R1': '17',
                                     'R2': '16',
                                     'R3': '4',
                                     'R4': '5',
                                     'R5': '20',
                                     'R6': '18'}},
                         {'source_row': 2,
                          'values': {'Plant': 'P3',
                                     'R1': '21',
                                     'R2': '19',
                                     'R3': '18',
                                     'R4': '17',
                                     'R5': '3',
                                     'R6': '4'}}],
             'returned_rows': 3,
             'role': 'route variable costs matrix',
             'table_id': 'file_2_view_0'},
            {'columns': ['Plant', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6'],
             'file_index': 3,
             'file_name': 'route_fixed_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 3,
             'records': [{'source_row': 0,
                          'values': {'Plant': 'P1',
                                     'R1': '260',
                                     'R2': '300',
                                     'R3': '620',
                                     'R4': '670',
                                     'R5': '710',
                                     'R6': '660'}},
                         {'source_row': 1,
                          'values': {'Plant': 'P2',
                                     'R1': '610',
                                     'R2': '590',
                                     'R3': '280',
                                     'R4': '310',
                                     'R5': '690',
                                     'R6': '640'}},
                         {'source_row': 2,
                          'values': {'Plant': 'P3',
                                     'R1': '700',
                                     'R2': '650',
                                     'R3': '630',
                                     'R4': '600',
                                     'R5': '250',
                                     'R6': '290'}}],
             'returned_rows': 3,
             'role': 'route fixed costs matrix',
             'table_id': 'file_3_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 6],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 6]},
                                  {'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [3, 6],
                                   'matrix_table_id': 'file_3_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'exact',
                                   'shape': [3, 6]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
import pandas as pd

def solve_problem(CSVQA_FRAMES):
    plant_frame = CSVQA_FRAMES['file_0_view_0']
    retailer_frame = CSVQA_FRAMES['file_1_view_0']
    I = [row['Plant'] for (_, row) in plant_frame.iterrows()]
    J = [row['Retailer'] for (_, row) in retailer_frame.iterrows()]
    SupplyCapacity = {}
    for (_, row) in plant_frame.iterrows():
        plant = row['Plant']
        try:
            SupplyCapacity[plant] = float(row['SupplyCapacity'])
        except Exception:
            raise ValueError(f"Invalid SupplyCapacity for plant {plant}: {row['SupplyCapacity']}")
    Demand = {}
    for (_, row) in retailer_frame.iterrows():
        retailer = row['Retailer']
        try:
            Demand[retailer] = float(row['Demand'])
        except Exception:
            raise ValueError(f"Invalid Demand for retailer {retailer}: {row['Demand']}")
    c_ij = {}
    varcost_frame = CSVQA_FRAMES['file_2_view_0']
    for (_, row) in varcost_frame.iterrows():
        plant = row['Plant']
        for retailer in J:
            try:
                c_ij[plant, retailer] = float(row[retailer])
            except Exception:
                raise ValueError(f'Invalid variable cost for ({plant},{retailer}): {row[retailer]}')
    f_ij = {}
    fixedcost_frame = CSVQA_FRAMES['file_3_view_0']
    for (_, row) in fixedcost_frame.iterrows():
        plant = row['Plant']
        for retailer in J:
            try:
                f_ij[plant, retailer] = float(row[retailer])
            except Exception:
                raise ValueError(f'Invalid fixed cost for ({plant},{retailer}): {row[retailer]}')
    M_ij = {}
    for i in I:
        for j in J:
            M_ij[i, j] = min(SupplyCapacity[i], Demand[j])
    for i in I:
        if i not in SupplyCapacity:
            raise KeyError(f'Missing SupplyCapacity for plant {i}')
    for j in J:
        if j not in Demand:
            raise KeyError(f'Missing Demand for retailer {j}')
    for i in I:
        for j in J:
            if (i, j) not in c_ij:
                raise KeyError(f'Missing variable cost for ({i},{j})')
            if (i, j) not in f_ij:
                raise KeyError(f'Missing fixed cost for ({i},{j})')
            if (i, j) not in M_ij:
                raise KeyError(f'Missing M_ij for ({i},{j})')
    m = gp.Model('FixedChargeTransportation')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars([(i, j) for i in I for j in J], lb=0, vtype=gp.GRB.CONTINUOUS, name='')
    y_vars = m.addVars([(i, j) for i in I for j in J], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] + f_ij[i, j] * y_vars[i, j] for i in I for j in J)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x_vars[i, j] for i in I)) == Demand[j] for j in J), name='')
    m.addConstrs((gp.quicksum((x_vars[i, j] for j in J)) <= SupplyCapacity[i] for i in I), name='')
    m.addConstrs((x_vars[i, j] <= M_ij[i, j] * y_vars[i, j] for i in I for j in J), name='')
    m.optimize()
    return m
m = solve_problem(CSVQA_FRAMES)