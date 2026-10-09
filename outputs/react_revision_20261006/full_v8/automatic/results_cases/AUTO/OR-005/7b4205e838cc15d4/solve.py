CSVQA_DATA = {'ignored_file_indices': [],
 'query': 'Amazon runs several distribution centers that deliver essential goods daily to different customer groups. '
          'The daily demand for each customer group is listed in “customer_demand.csv,” while the daily supply '
          'capacity of the distribution centers is outlined in “supply_capacity.csv.” The transportation cost per unit '
          'from each distribution center to each customer group is specified in “transportation_costs.csv.” The goal '
          'is to decide the quantity of goods to be shipped from each distribution center to each customer group, '
          'ensuring all demands are fulfilled without exceeding the supply capacity of any center, while minimizing '
          'the total transportation cost.',
 'relationships': [{'column_axis': {'id_column': 'Customers', 'table_id': 'file_0_view_0'},
                    'matrix_table_id': 'file_2_view_0',
                    'row_axis': {'id_column': 'Supplier', 'table_id': 'file_1_view_0'},
                    'row_id_column': 'Unnamed: 0',
                    'row_id_mapping': {'supply1': 'supplier1',
                                       'supply2': 'supplier2',
                                       'supply3': 'supplier3',
                                       'supply4': 'supplier4',
                                       'supply5': 'supplier5',
                                       'supply6': 'supplier6',
                                       'supply7': 'supplier7',
                                       'supply8': 'supplier8'},
                    'type': 'matrix'}],
 'route': 'TP',
 'tables': [{'columns': ['Customers', 'demand'],
             'file_index': 0,
             'file_name': 'customer_demand.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Customers': 'demand1', 'demand': '9'}},
                         {'source_row': 1, 'values': {'Customers': 'demand2', 'demand': '66'}},
                         {'source_row': 2, 'values': {'Customers': 'demand3', 'demand': '56'}},
                         {'source_row': 3, 'values': {'Customers': 'demand4', 'demand': '17'}},
                         {'source_row': 4, 'values': {'Customers': 'demand5', 'demand': '43'}},
                         {'source_row': 5, 'values': {'Customers': 'demand6', 'demand': '62'}},
                         {'source_row': 6, 'values': {'Customers': 'demand7', 'demand': '10'}},
                         {'source_row': 7, 'values': {'Customers': 'demand8', 'demand': '37'}}],
             'returned_rows': 8,
             'role': 'customer demand',
             'table_id': 'file_0_view_0'},
            {'columns': ['Supplier', 'supply_capacity'],
             'file_index': 1,
             'file_name': 'supply_capacity.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0, 'values': {'Supplier': 'supplier1', 'supply_capacity': '60'}},
                         {'source_row': 1, 'values': {'Supplier': 'supplier2', 'supply_capacity': '22'}},
                         {'source_row': 2, 'values': {'Supplier': 'supplier3', 'supply_capacity': '16'}},
                         {'source_row': 3, 'values': {'Supplier': 'supplier4', 'supply_capacity': '14'}},
                         {'source_row': 4, 'values': {'Supplier': 'supplier5', 'supply_capacity': '19'}},
                         {'source_row': 5, 'values': {'Supplier': 'supplier6', 'supply_capacity': '70'}},
                         {'source_row': 6, 'values': {'Supplier': 'supplier7', 'supply_capacity': '60'}},
                         {'source_row': 7, 'values': {'Supplier': 'supplier8', 'supply_capacity': '39'}}],
             'returned_rows': 8,
             'role': 'supply capacity',
             'table_id': 'file_1_view_0'},
            {'columns': ['Unnamed: 0',
                         'demand1',
                         'demand2',
                         'demand3',
                         'demand4',
                         'demand5',
                         'demand6',
                         'demand7',
                         'demand8'],
             'file_index': 2,
             'file_name': 'transportation_costs.csv',
             'filters': {'conditions': [], 'logic': 'and'},
             'original_rows': 8,
             'records': [{'source_row': 0,
                          'values': {'Unnamed: 0': 'supply1',
                                     'demand1': '0.03020736643461065',
                                     'demand2': '229.50723504640203',
                                     'demand3': '198.62356558205792',
                                     'demand4': '12.995050640153751',
                                     'demand5': '211.20732124396406',
                                     'demand6': '134.9442985029274',
                                     'demand7': '9.822206398831067',
                                     'demand8': '11.394077543225675'}},
                         {'source_row': 1,
                          'values': {'Unnamed: 0': 'supply2',
                                     'demand1': '232.34691308087835',
                                     'demand2': '3.6258726438627473',
                                     'demand3': '0.28605434149404785',
                                     'demand4': '45.73127693242935',
                                     'demand5': '2.8304796563034573',
                                     'demand6': '107.05891033185472',
                                     'demand7': '299.96317913389305',
                                     'demand8': '23.79935436307657'}},
                         {'source_row': 2,
                          'values': {'Unnamed: 0': 'supply3',
                                     'demand1': '11.061938334356302',
                                     'demand2': '0.2041995326579051',
                                     'demand3': '0.2789447278030927',
                                     'demand4': '45.721912724349636',
                                     'demand5': '59.54895565737313',
                                     'demand6': '5.097536739581239',
                                     'demand7': '300.00118415135785',
                                     'demand8': '23.711282707746893'}},
                         {'source_row': 3,
                          'values': {'Unnamed: 0': 'supply4',
                                     'demand1': '235.1794835706472',
                                     'demand2': '43.794668963036194',
                                     'demand3': '40.709846782945924',
                                     'demand4': '0.07774496620087613',
                                     'demand5': '4.237728183419554',
                                     'demand6': '131.70915517494691',
                                     'demand7': '296.55587567706743',
                                     'demand8': '29.810940017561297'}},
                         {'source_row': 4,
                          'values': {'Unnamed: 0': 'supply5',
                                     'demand1': '211.85808746383796',
                                     'demand2': '47.60180876530328',
                                     'demand3': '50.04007716193931',
                                     'demand4': '86.14548807358399',
                                     'demand5': '0.06197897916874956',
                                     'demand6': '5.3345515296262205',
                                     'demand7': '270.06290423798396',
                                     'demand8': '3.853933133973331'}},
                         {'source_row': 5,
                          'values': {'Unnamed: 0': 'supply6',
                                     'demand1': '6.45506633554524',
                                     'demand2': '88.16323623354015',
                                     'demand3': '5.047091671641611',
                                     'demand4': '151.46120287365497',
                                     'demand5': '5.290760161059401',
                                     'demand6': '0.04602205335871525',
                                     'demand7': '9.93670660180487',
                                     'demand8': '103.75460989446313'}},
                         {'source_row': 6,
                          'values': {'Unnamed: 0': 'supply7',
                                     'demand1': '174.27229047340035',
                                     'demand2': '250.58223528739327',
                                     'demand3': '253.90413041857263',
                                     'demand4': '16.235467318386764',
                                     'demand5': '12.643140514778086',
                                     'demand6': '175.0672824108511',
                                     'demand7': '2.983839625303656',
                                     'demand8': '317.0655193866389'}},
                         {'source_row': 7,
                          'values': {'Unnamed: 0': 'supply8',
                                     'demand1': '207.87006253790491',
                                     'demand2': '1.517168471518212',
                                     'demand3': '24.027239288137153',
                                     'demand4': '27.133999276450346',
                                     'demand5': '73.20672468851855',
                                     'demand6': '125.72910359893308',
                                     'demand7': '15.463103251642147',
                                     'demand8': '0.20164987511903337'}}],
             'returned_rows': 8,
             'role': 'transportation cost matrix',
             'table_id': 'file_2_view_0'}],
 'validation': {'matrix_checks': [{'column_ids_aligned': True,
                                   'column_mapping_basis': 'exact',
                                   'expected_shape': [8, 8],
                                   'matrix_table_id': 'file_2_view_0',
                                   'row_ids_aligned': True,
                                   'row_mapping_basis': 'unique_complete_suffix',
                                   'shape': [8, 8]}],
                'status': 'OK'}}
import pandas as pd
CSVQA_FRAMES = {t["table_id"]: pd.DataFrame([r["values"] for r in t["records"]], columns=t["columns"], index=[r["source_row"] for r in t["records"]]) for t in CSVQA_DATA["tables"]}
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    # Read data from CSVQA_FRAMES
    # Customers and their demands
    customer_frame = CSVQA_FRAMES["file_0_view_0"]
    customers = []
    demand = {}
    for _, row in customer_frame.iterrows():
        cust = row["Customers"]
        customers.append(cust)
        try:
            demand[cust] = float(row["demand"])
        except Exception:
            raise ValueError(f"Invalid demand value for customer {cust}: {row['demand']}")

    # Suppliers and their supply capacities
    supplier_frame = CSVQA_FRAMES["file_1_view_0"]
    suppliers = []
    supply_capacity = {}
    for _, row in supplier_frame.iterrows():
        sup = row["Supplier"]
        suppliers.append(sup)
        try:
            supply_capacity[sup] = float(row["supply_capacity"])
        except Exception:
            raise ValueError(f"Invalid supply_capacity value for supplier {sup}: {row['supply_capacity']}")

    # Transportation cost matrix
    cost_frame = CSVQA_FRAMES["file_2_view_0"]
    # Apply row_id_mapping
    row_id_mapping = {
        "supply1": "supplier1",
        "supply2": "supplier2",
        "supply3": "supplier3",
        "supply4": "supplier4",
        "supply5": "supplier5",
        "supply6": "supplier6",
        "supply7": "supplier7",
        "supply8": "supplier8"
    }
    cost = {}
    for _, row in cost_frame.iterrows():
        raw_row_id = row["Unnamed: 0"]
        if raw_row_id not in row_id_mapping:
            raise ValueError(f"Row id {raw_row_id} not in row_id_mapping")
        sup = row_id_mapping[raw_row_id]
        if sup not in suppliers:
            raise ValueError(f"Supplier {sup} from cost matrix not in supplier list")
        cost[sup] = {}
        for cust in customers:
            if cust not in row:
                raise ValueError(f"Customer {cust} not found in cost matrix columns")
            try:
                cost[sup][cust] = float(row[cust])
            except Exception:
                raise ValueError(f"Invalid cost value for supplier {sup}, customer {cust}: {row[cust]}")

    # Validate dimensions
    if set(cost.keys()) != set(suppliers):
        raise ValueError("Mismatch between suppliers in cost matrix and supplier list")
    for sup in suppliers:
        if set(cost[sup].keys()) != set(customers):
            raise ValueError(f"Mismatch between customers in cost matrix and customer list for supplier {sup}")

    # Build model
    m = gp.Model("Amazon_Distribution_TP")
    m.Params.MIPGap = 1e-4

    # Decision variables: x_{ij} >= 0, continuous
    quantity_vars = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name="x")

    # Objective: minimize total transportation cost
    m.setObjective(
        gp.quicksum(cost[i][j] * quantity_vars[i, j] for i in suppliers for j in customers),
        GRB.MINIMIZE
    )

    # Demand satisfaction: sum over suppliers >= demand for each customer
    m.addConstrs(
        (gp.quicksum(quantity_vars[i, j] for i in suppliers) >= demand[j] for j in customers),
        name="demand"
    )

    # Supply capacity: sum over customers <= supply_capacity for each supplier
    m.addConstrs(
        (gp.quicksum(quantity_vars[i, j] for j in customers) <= supply_capacity[i] for i in suppliers),
        name="supply"
    )

    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for variable in m.getVars():
            print(f'{variable.VarName}: {variable.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m

m = solve_problem()