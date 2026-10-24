CSVQA_DATA = {'query': 'Amazon runs several distribution centers that deliver essential goods daily to different customer groups. The daily demand for each customer group is listed in “customer_demand.csv,” while the daily supply capacity of the distribution centers is outlined in “supply_capacity.csv.” The transportation cost per unit from each distribution center to each customer group is specified in “transportation_costs.csv.” The goal is to decide the quantity of goods to be shipped from each distribution center to each customer group, ensuring all demands are fulfilled without exceeding the supply capacity of any center, while minimizing the total transportation cost.', 'relationships': [], 'tables': [{'columns': ['Customers', 'demand'], 'file_name': 'customer_demand.csv', 'records': [{'data_row_number': 0, 'values': {'Customers': 'demand1', 'demand': '9'}}, {'data_row_number': 1, 'values': {'Customers': 'demand2', 'demand': '66'}}, {'data_row_number': 2, 'values': {'Customers': 'demand3', 'demand': '56'}}, {'data_row_number': 3, 'values': {'Customers': 'demand4', 'demand': '17'}}, {'data_row_number': 4, 'values': {'Customers': 'demand5', 'demand': '43'}}, {'data_row_number': 5, 'values': {'Customers': 'demand6', 'demand': '62'}}, {'data_row_number': 6, 'values': {'Customers': 'demand7', 'demand': '10'}}, {'data_row_number': 7, 'values': {'Customers': 'demand8', 'demand': '37'}}], 'table_id': 'file_0_view_0'}, {'columns': ['Supplier', 'supply_capacity'], 'file_name': 'supply_capacity.csv', 'records': [{'data_row_number': 0, 'values': {'Supplier': 'supplier1', 'supply_capacity': '60'}}, {'data_row_number': 1, 'values': {'Supplier': 'supplier2', 'supply_capacity': '22'}}, {'data_row_number': 2, 'values': {'Supplier': 'supplier3', 'supply_capacity': '16'}}, {'data_row_number': 3, 'values': {'Supplier': 'supplier4', 'supply_capacity': '14'}}, {'data_row_number': 4, 'values': {'Supplier': 'supplier5', 'supply_capacity': '19'}}, {'data_row_number': 5, 'values': {'Supplier': 'supplier6', 'supply_capacity': '70'}}, {'data_row_number': 6, 'values': {'Supplier': 'supplier7', 'supply_capacity': '60'}}, {'data_row_number': 7, 'values': {'Supplier': 'supplier8', 'supply_capacity': '39'}}], 'table_id': 'file_1_view_0'}, {'columns': ['Unnamed: 0', 'demand1', 'demand2', 'demand3', 'demand4', 'demand5', 'demand6', 'demand7', 'demand8'], 'file_name': 'transportation_costs.csv', 'records': [{'data_row_number': 0, 'values': {'Unnamed: 0': 'supply1', 'demand1': '0.03020736643461065', 'demand2': '229.50723504640203', 'demand3': '198.62356558205792', 'demand4': '12.995050640153751', 'demand5': '211.20732124396406', 'demand6': '134.9442985029274', 'demand7': '9.822206398831067', 'demand8': '11.394077543225675'}}, {'data_row_number': 1, 'values': {'Unnamed: 0': 'supply2', 'demand1': '232.34691308087835', 'demand2': '3.6258726438627473', 'demand3': '0.28605434149404785', 'demand4': '45.73127693242935', 'demand5': '2.8304796563034573', 'demand6': '107.05891033185472', 'demand7': '299.96317913389305', 'demand8': '23.79935436307657'}}, {'data_row_number': 2, 'values': {'Unnamed: 0': 'supply3', 'demand1': '11.061938334356302', 'demand2': '0.2041995326579051', 'demand3': '0.2789447278030927', 'demand4': '45.721912724349636', 'demand5': '59.54895565737313', 'demand6': '5.097536739581239', 'demand7': '300.00118415135785', 'demand8': '23.711282707746893'}}, {'data_row_number': 3, 'values': {'Unnamed: 0': 'supply4', 'demand1': '235.1794835706472', 'demand2': '43.794668963036194', 'demand3': '40.709846782945924', 'demand4': '0.07774496620087613', 'demand5': '4.237728183419554', 'demand6': '131.70915517494691', 'demand7': '296.55587567706743', 'demand8': '29.810940017561297'}}, {'data_row_number': 4, 'values': {'Unnamed: 0': 'supply5', 'demand1': '211.85808746383796', 'demand2': '47.60180876530328', 'demand3': '50.04007716193931', 'demand4': '86.14548807358399', 'demand5': '0.06197897916874956', 'demand6': '5.3345515296262205', 'demand7': '270.06290423798396', 'demand8': '3.853933133973331'}}, {'data_row_number': 5, 'values': {'Unnamed: 0': 'supply6', 'demand1': '6.45506633554524', 'demand2': '88.16323623354015', 'demand3': '5.047091671641611', 'demand4': '151.46120287365497', 'demand5': '5.290760161059401', 'demand6': '0.04602205335871525', 'demand7': '9.93670660180487', 'demand8': '103.75460989446313'}}, {'data_row_number': 6, 'values': {'Unnamed: 0': 'supply7', 'demand1': '174.27229047340035', 'demand2': '250.58223528739327', 'demand3': '253.90413041857263', 'demand4': '16.235467318386764', 'demand5': '12.643140514778086', 'demand6': '175.0672824108511', 'demand7': '2.983839625303656', 'demand8': '317.0655193866389'}}, {'data_row_number': 7, 'values': {'Unnamed: 0': 'supply8', 'demand1': '207.87006253790491', 'demand2': '1.517168471518212', 'demand3': '24.027239288137153', 'demand4': '27.133999276450346', 'demand5': '73.20672468851855', 'demand6': '125.72910359893308', 'demand7': '15.463103251642147', 'demand8': '0.20164987511903337'}}], 'table_id': 'file_2_view_0'}]}
import gurobipy
from gurobipy import GRB

def solve_problem():
    tables = CSVQA_DATA['tables']
    table0 = next((t for t in tables if t['table_id'] == 'file_0_view_0'))
    table1 = next((t for t in tables if t['table_id'] == 'file_1_view_0'))
    table2 = next((t for t in tables if t['table_id'] == 'file_2_view_0'))
    customers = [rec['values']['Customers'] for rec in table0['records']]
    demand_map = {}
    for rec in table0['records']:
        key = rec['values']['Customers']
        val = rec['values']['demand']
        try:
            demand_map[key] = float(val)
        except ValueError:
            raise ValueError(f'Invalid demand value for {key}')
    original_suppliers = [rec['values']['Supplier'] for rec in table1['records']]
    supply_capacity_map = {}
    for rec in table1['records']:
        key = rec['values']['Supplier']
        val = rec['values']['supply_capacity']
        try:
            supply_capacity_map[key] = float(val)
        except ValueError:
            raise ValueError(f'Invalid supply capacity for {key}')
    cost_suppliers = [rec['values']['Unnamed: 0'] for rec in table2['records']]
    if len(cost_suppliers) != len(original_suppliers):
        raise ValueError('Supplier count mismatch between cost matrix and capacity table')
    cost_to_original = dict(zip(cost_suppliers, original_suppliers))
    capacity = {}
    for cs in cost_suppliers:
        orig = cost_to_original[cs]
        if orig not in supply_capacity_map:
            raise ValueError(f'Missing supply capacity for supplier {orig}')
        capacity[cs] = supply_capacity_map[orig]
    cost_map = {}
    for rec in table2['records']:
        supplier_id = rec['values']['Unnamed: 0']
        for cust in customers:
            if cust not in rec['values']:
                raise ValueError(f'Missing cost for supplier {supplier_id} to customer {cust}')
            val = rec['values'][cust]
            try:
                cost_map[supplier_id, cust] = float(val)
            except ValueError:
                raise ValueError(f'Invalid cost value for supplier {supplier_id} to customer {cust}')
    for i in cost_suppliers:
        for j in customers:
            if (i, j) not in cost_map:
                raise ValueError(f'Missing cost for supplier {i} to customer {j}')
    m = gurobipy.Model('TransportationProblem')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars([(i, j) for i in cost_suppliers for j in customers], lb=0, vtype=GRB.CONTINUOUS, name='')
    m.setObjective(gurobipy.quicksum((cost_map[i, j] * x[i, j] for i in cost_suppliers for j in customers)), GRB.MINIMIZE)
    m.addConstrs((gurobipy.quicksum((x[i, j] for i in cost_suppliers)) >= demand_map[j] for j in customers), name='')
    m.addConstrs((gurobipy.quicksum((x[i, j] for j in customers)) <= capacity[i] for i in cost_suppliers), name='')
    m.optimize()
    return m
m = solve_problem()
