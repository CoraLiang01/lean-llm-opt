CSVQA_DATA = {'query': 'multiple supermarkets require inventory, and several suppliers located in different cities can provide the necessary goods. Each supplier incurs a fixed cost upon starting operations, with the fixed cost data provided in "fixed_cost.csv." Each supermarket needs to source a unit good from these suppliers. For each supermarket, the transportation cost per unit of goods from each supplier is recorded in "transportation_costs.csv." Demand information can be gained in \'demand.csv\'. The objective is to determine which suppliers to open so that the demand of all supermarkets is met while minimizing the total cost. The decision variables y_i are binary, indicating whether a supplier is operational (open). The decision variables x_ij represent the quantity of goods that supermarket S_j sources from supplier F_i. For each supermarket, x_ij represents the proportion of the total supply obtained from different suppliers. These decision variables help determine the optimal allocation of supply to minimize the total of fixed and transportation costs.', 'relationships': [], 'tables': [{'columns': ['customer', 'demand'], 'file_name': 'demand.csv', 'records': [{'data_row_number': 0, 'values': {'customer': 'C1', 'demand': '1097'}}, {'data_row_number': 1, 'values': {'customer': 'C2', 'demand': '61'}}, {'data_row_number': 2, 'values': {'customer': 'C3', 'demand': '11'}}, {'data_row_number': 3, 'values': {'customer': 'C4', 'demand': '7'}}, {'data_row_number': 4, 'values': {'customer': 'C5', 'demand': '82'}}, {'data_row_number': 5, 'values': {'customer': 'C6', 'demand': '37'}}, {'data_row_number': 6, 'values': {'customer': 'C7', 'demand': '483'}}, {'data_row_number': 7, 'values': {'customer': 'C8', 'demand': '582'}}, {'data_row_number': 8, 'values': {'customer': 'C9', 'demand': '223'}}, {'data_row_number': 9, 'values': {'customer': 'C10', 'demand': '89'}}, {'data_row_number': 10, 'values': {'customer': 'C11', 'demand': '60'}}, {'data_row_number': 11, 'values': {'customer': 'C12', 'demand': '55'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Unnamed: 0', 'fixed_costs'], 'file_name': 'fixed_cost.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'S1', 'fixed_costs': '98.88'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'S2', 'fixed_costs': '99.73'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'S3', 'fixed_costs': '94.01000000000001'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'S4', 'fixed_costs': '93.77'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'S5', 'fixed_costs': '107.59'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'S6', 'fixed_costs': '112.65'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'S7', 'fixed_costs': '97.05'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'S8', 'fixed_costs': '103'}}, {'data_row_number': 8, 'values': {'Unnamed: 0': 'S9', 'fixed_costs': '90.45'}}, {'data_row_number': 9, 'values': {'Unnamed: 0': 'S10', 'fixed_costs': '96.73'}}, {'data_row_number': 10, 'values': {'Unnamed: 0': 'S11', 'fixed_costs': '96.43000000000001'}}, {'data_row_number': 11, 'values': {'Unnamed: 0': 'S12', 'fixed_costs': '112.19'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'C1': '284.11', 'C10': '8.869999999999999', 'C11': '1129.47', 'C12': '185.53', 'C2': '53.78', 'C3': '10.62', 'C4': '111.27', 'C5': '158.5', 'C6': '8.789999999999999', 'C7': '53.79', 'C8': '8.84', 'C9': '1911.43', 'Unnamed: 0': 'S1'}}, {'data_row_number': 1, 'values': {'C1': '7.19', 'C10': '1.45', 'C11': '49.14', 'C12': '0.05', 'C2': '1031.96', 'C3': '90.94', 'C4': '276.97', 'C5': '0.45', 'C6': '0.2', 'C7': '49.14', 'C8': '1.05', 'C9': '2079.54', 'Unnamed: 0': 'S2'}}, {'data_row_number': 2, 'values': {'C1': '151.1', 'C10': '1.63', 'C11': '884.47', 'C12': '0.96', 'C2': '884.48', 'C3': '4.33', 'C4': '277.04', 'C5': '0.33', 'C6': '0.19', 'C7': '49.14', 'C8': '0.99', 'C9': '99.03', 'Unnamed: 0': 'S3'}}, {'data_row_number': 3, 'values': {'C1': '144.16', 'C10': '19.74', 'C11': '868.74', 'C12': '19.85', 'C2': '868.75', 'C3': '94.2', 'C4': '285.48', 'C5': '16.93', 'C6': '0.9399999999999999', 'C7': '868.78', 'C8': '16.6', 'C9': '98.69', 'Unnamed: 0': 'S4'}}, {'data_row_number': 4, 'values': {'C1': '151.34', 'C10': '0.84', 'C11': '883.6', 'C12': '0.58', 'C2': '1030.88', 'C3': '91.43000000000001', 'C4': '13.24', 'C5': '0.72', 'C6': '0.87', 'C7': '49.09', 'C8': '0.01', 'C9': '99.05', 'Unnamed: 0': 'S5'}}, {'data_row_number': 5, 'values': {'C1': '7.18', 'C10': '1.06', 'C11': '884.3099999999999', 'C12': '0.34', 'C2': '49.13', 'C3': '90.72', 'C4': '277.57', 'C5': '0.37', 'C6': '0.58', 'C7': '1031.74', 'C8': '0.76', 'C9': '1782.98', 'Unnamed: 0': 'S6'}}, {'data_row_number': 6, 'values': {'C1': '104.38', 'C10': '2080.99', 'C11': '1545.08', 'C12': '99.06999999999999', 'C2': '1324.35', 'C3': '1829.39', 'C4': '1857.57', 'C5': '1782.69', 'C6': '2079.47', 'C7': '1324.31', 'C8': '2080.29', 'C9': '0', 'Unnamed: 0': 'S7'}}, {'data_row_number': 7, 'values': {'C1': '129.51', 'C10': '1.69', 'C11': '49.14', 'C12': '0.05', 'C2': '1031.96', 'C3': '4.33', 'C4': '276.97', 'C5': '0.02', 'C6': '0.23', 'C7': '884.5599999999999', 'C8': '1.22', 'C9': '2079.54', 'Unnamed: 0': 'S8'}}, {'data_row_number': 8, 'values': {'C1': '50.93', 'C10': '47.59', 'C11': '5.75', 'C12': '999.9400000000001', 'C2': '5.75', 'C3': '1057.85', 'C4': '58.62', 'C5': '47.63', 'C6': '1000.41', 'C7': '103.48', 'C8': '47.6', 'C9': '1642.85', 'Unnamed: 0': 'S9'}}, {'data_row_number': 9, 'values': {'C1': '129.62', 'C10': '0.08', 'C11': '49.13', 'C12': '0.04', 'C2': '884.35', 'C3': '91.09999999999999', 'C4': '277.12', 'C5': '0.27', 'C6': '0.07000000000000001', 'C7': '1031.78', 'C8': '0.91', 'C9': '99.03', 'Unnamed: 0': 'S10'}}, {'data_row_number': 10, 'values': {'C1': '53.3', 'C10': '49.1', 'C11': '0.08', 'C12': '49.12', 'C2': '0', 'C3': '941.91', 'C4': '58.92', 'C5': '1031.61', 'C6': '49.13', 'C7': '0.03', 'C8': '1030.99', 'C9': '1324.29', 'Unnamed: 0': 'S11'}}, {'data_row_number': 11, 'values': {'C1': '959.55', 'C10': '49.1', 'C11': '0.12', 'C12': '1031.53', 'C2': '0.11', 'C3': '941.98', 'C4': '1237.42', 'C5': '49.13', 'C6': '1031.86', 'C7': '0.09', 'C8': '1031.07', 'C9': '73.56999999999999', 'Unnamed: 0': 'S12'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']

    def get_table(table_id):
        for t in tables:
            if t['table_id'] == table_id:
                return t
        raise ValueError(f'Table {table_id} not found')
    cust_table = get_table('file_0_view_0')
    J = []
    d_j = {}
    for rec in cust_table['records']:
        cust = rec['values']['customer']
        J.append(cust)
        d_j[cust] = float(rec['values']['demand'])
    supp_table = get_table('file_1_view_0')
    I = []
    f_i = {}
    for rec in supp_table['records']:
        supp = rec['values']['Unnamed: 0']
        I.append(supp)
        f_i[supp] = float(rec['values']['fixed_costs'])
    cost_table = get_table('file_2_view_0')
    cost_cols = [c for c in cost_table['columns'] if c != 'Unnamed: 0']
    missing_cols = set(J) - set(cost_cols)
    if missing_cols:
        raise ValueError(f'Missing cost columns for customers: {missing_cols}')
    c = {}
    for rec in cost_table['records']:
        supp = rec['values']['Unnamed: 0']
        if supp not in I:
            raise ValueError(f'Supplier {supp} in cost table not in supplier list')
        for cust in J:
            key = (supp, cust)
            try:
                c[key] = float(rec['values'][cust])
            except KeyError:
                raise ValueError(f'Missing cost for supplier {supp} to customer {cust}')
    for i in I:
        for j in J:
            if (i, j) not in c:
                raise ValueError(f'Cost missing for pair {i},{j}')
    model = gurobipy.Model()
    model.setParam('MIPGap', 1e-06)
    y = model.addVars(I, vtype=GRB.BINARY, name='')
    x = model.addVars([(i, j) for i in I for j in J], vtype=GRB.CONTINUOUS, lb=0, name='')
    model.addConstrs((gurobipy.quicksum((x[i, j] for i in I)) >= d_j[j] for j in J), name='')
    model.addConstrs((x[i, j] <= d_j[j] * y[i] for i in I for j in J), name='')
    obj = gurobipy.quicksum((f_i[i] * y[i] for i in I)) + gurobipy.quicksum((c[i, j] * x[i, j] for i in I for j in J))
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    return model
m = solve_problem()
