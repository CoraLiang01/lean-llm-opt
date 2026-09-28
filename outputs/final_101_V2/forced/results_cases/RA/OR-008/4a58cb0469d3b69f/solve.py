LEGACY_OBSERVATION = '{"values": {"Customers": "Customer1", "demand": "70"}}\n{"values": {"Customers": "Customer2", "demand": "80"}}\n{"values": {"Customers": "Customer3", "demand": "60"}}\n{"values": {"Customers": "Customer4", "demand": "90"}}\n{"values": {"Customers": "Customer5", "demand": "85"}}\n{"values": {"Customers": "Customer6", "demand": "95"}}\n{"values": {"Suppliers": "Supplier1", "supply_capacity": "200"}}\n{"values": {"Suppliers": "Supplier2", "supply_capacity": "250"}}\n{"values": {"Suppliers": "Supplier3", "supply_capacity": "230"}}\n{"values": {"Suppliers": "Supplier4", "supply_capacity": "220"}}\n{"values": {"Suppliers": "Supplier5", "supply_capacity": "210"}}\n{"values": {"Unnamed: 0": "Supplier1", "Customer1": "2", "Customer2": "3", "Customer3": "1", "Customer4": "2", "Customer5": "3", "Customer6": "2"}}\n{"values": {"Unnamed: 0": "Supplier2", "Customer1": "1", "Customer2": "2", "Customer3": "3", "Customer4": "2", "Customer5": "3", "Customer6": "2"}}\n{"values": {"Unnamed: 0": "Supplier3", "Customer1": "3", "Customer2": "1", "Customer3": "2", "Customer4": "3", "Customer5": "2", "Customer6": "3"}}\n{"values": {"Unnamed: 0": "Supplier4", "Customer1": "2", "Customer2": "3", "Customer3": "2", "Customer4": "1", "Customer5": "3", "Customer6": "4"}}\n{"values": {"Unnamed: 0": "Supplier5", "Customer1": "3", "Customer2": "2", "Customer3": "3", "Customer4": "3", "Customer5": "2", "Customer6": "3"}}'
LEGACY_RECORDS = [{'source': '', 'values': {'Customers': 'Customer1', 'demand': '70'}}, {'source': '', 'values': {'Customers': 'Customer2', 'demand': '80'}}, {'source': '', 'values': {'Customers': 'Customer3', 'demand': '60'}}, {'source': '', 'values': {'Customers': 'Customer4', 'demand': '90'}}, {'source': '', 'values': {'Customers': 'Customer5', 'demand': '85'}}, {'source': '', 'values': {'Customers': 'Customer6', 'demand': '95'}}, {'source': '', 'values': {'Suppliers': 'Supplier1', 'supply_capacity': '200'}}, {'source': '', 'values': {'Suppliers': 'Supplier2', 'supply_capacity': '250'}}, {'source': '', 'values': {'Suppliers': 'Supplier3', 'supply_capacity': '230'}}, {'source': '', 'values': {'Suppliers': 'Supplier4', 'supply_capacity': '220'}}, {'source': '', 'values': {'Suppliers': 'Supplier5', 'supply_capacity': '210'}}, {'source': '', 'values': {'Unnamed: 0': 'Supplier1', 'Customer1': '2', 'Customer2': '3', 'Customer3': '1', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2'}}, {'source': '', 'values': {'Unnamed: 0': 'Supplier2', 'Customer1': '1', 'Customer2': '2', 'Customer3': '3', 'Customer4': '2', 'Customer5': '3', 'Customer6': '2'}}, {'source': '', 'values': {'Unnamed: 0': 'Supplier3', 'Customer1': '3', 'Customer2': '1', 'Customer3': '2', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3'}}, {'source': '', 'values': {'Unnamed: 0': 'Supplier4', 'Customer1': '2', 'Customer2': '3', 'Customer3': '2', 'Customer4': '1', 'Customer5': '3', 'Customer6': '4'}}, {'source': '', 'values': {'Unnamed: 0': 'Supplier5', 'Customer1': '3', 'Customer2': '2', 'Customer3': '3', 'Customer4': '3', 'Customer5': '2', 'Customer6': '3'}}]
import gurobipy as gp
from gurobipy import GRB
customers = []
demand = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'Customers' in v and 'demand' in v:
        cust = v['Customers']
        customers.append(cust)
        demand[cust] = float(v['demand'])
suppliers = []
supply_capacity = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'Suppliers' in v and 'supply_capacity' in v:
        sup = v['Suppliers']
        suppliers.append(sup)
        supply_capacity[sup] = float(v['supply_capacity'])
cost = {}
for rec in LEGACY_RECORDS:
    v = rec['values']
    if 'Unnamed: 0' in v:
        sup = v['Unnamed: 0']
        cost[sup] = {}
        for cust in customers:
            if cust not in v:
                raise ValueError(f'Missing cost for {sup}, {cust}')
            cost[sup][cust] = float(v[cust])
for i in suppliers:
    if i not in cost:
        raise ValueError(f'Missing cost row for supplier {i}')
    for j in customers:
        if j not in cost[i]:
            raise ValueError(f'Missing cost for supplier {i}, customer {j}')
m = gp.Model('FreshMart_Transport')
x = m.addVars(suppliers, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((cost[i][j] * x[i, j] for i in suppliers for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in suppliers)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= supply_capacity[i] for i in suppliers), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')