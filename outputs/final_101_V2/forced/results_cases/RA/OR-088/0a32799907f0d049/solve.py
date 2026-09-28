LEGACY_OBSERVATION = 'cost.csv\n\nplant,fixed_cost,capacity,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15\nF1,11250,101,7.8,7.6,6.7,7.9,8.1,8.3,7.3,8.2,8.1,8.2,7.3,7.7,6.7,7.1,7.9\nF2,13480,124,5.3,6,5,6.4,5.9,6.2,5.6,6.1,6.3,6.1,5,5.6,5.3,4.9,6.3\nF3,14870,139,7.2,8.1,7.4,8.8,8.5,8.7,7.7,8.7,8.9,8.5,7.2,7.7,7.1,7.6,8.4\nF4,10290,86,7,7.1,6.5,7.9,7.4,7.7,6.7,7.9,7.8,7.3,6.8,7,6.5,6.7,7.6\nF5,16740,157,3.5,3.8,2.9,4.3,3.6,3.9,3.2,4.3,4.5,4,3.2,4,2.9,3.4,3.9\nF6,13960,133,8.2,8.6,7.9,9.5,8.5,9.3,8.5,9.4,9,9.2,8.1,8.7,7.9,8.5,9\nF7,12680,118,6.9,7.6,6.8,8.4,8,8,7.6,8,8.1,7.8,6.9,7.1,7,6.9,7.5\nF8,17890,162,6.9,7.8,7.1,8.7,8.6,8.2,7.2,7.9,8.4,7.9,7,7.4,6.8,7.3,8\nF9,10950,92,3.5,3.8,2.8,4.4,4.2,4.8,3.8,5,4.5,4.1,3.2,3.7,3.7,3.2,4.5\nF10,15320,144,5.2,6.1,5.1,6.3,6.1,6,5.6,6.5,6.2,5.9,5.3,6.1,5.1,5.2,6.2\nF11,11830,107,5.2,5.5,4.5,6.2,5.7,6.1,5.1,5.8,5.7,6.2,5.2,5.2,4.5,5.1,5.4\nF12,14110,129,7.8,8.7,7.6,9,8.6,9,8.5,9.3,9.3,8.4,7.9,8.2,7.4,7.6,8.7\nF13,15970,151,6.7,6.6,6.1,7.3,7.1,7.5,6.7,8,7.6,7.2,6.3,6.9,6.2,6,7.2\nF14,13140,113,7.5,8.6,7.6,8.2,8,7.9,7.5,8.7,8.8,8.1,7.2,7.3,7,7,8\nF15,10580,85,5.1,5.8,4.6,5.9,6.5,5.9,5.2,7,7.1,5.9,5.1,5.8,5.4,4.9,6\n\ndemand.csv\n\ncustomer,demand\nC1,83\nC2,76\nC3,91\nC4,68\nC5,104\nC6,97\nC7,88\nC8,73\nC9,109\nC10,95\nC11,82\nC12,67\nC13,113\nC14,79\nC15,92'
LEGACY_RECORDS = [{'source': 'cost.csv', 'values': {'plant': 'F1', 'fixed_cost': '11250', 'capacity': '101', 'C1': '7.8', 'C2': '7.6', 'C3': '6.7', 'C4': '7.9', 'C5': '8.1', 'C6': '8.3', 'C7': '7.3', 'C8': '8.2', 'C9': '8.1', 'C10': '8.2', 'C11': '7.3', 'C12': '7.7', 'C13': '6.7', 'C14': '7.1', 'C15': '7.9'}}, {'source': 'cost.csv', 'values': {'plant': 'F2', 'fixed_cost': '13480', 'capacity': '124', 'C1': '5.3', 'C2': '6', 'C3': '5', 'C4': '6.4', 'C5': '5.9', 'C6': '6.2', 'C7': '5.6', 'C8': '6.1', 'C9': '6.3', 'C10': '6.1', 'C11': '5', 'C12': '5.6', 'C13': '5.3', 'C14': '4.9', 'C15': '6.3'}}, {'source': 'cost.csv', 'values': {'plant': 'F3', 'fixed_cost': '14870', 'capacity': '139', 'C1': '7.2', 'C2': '8.1', 'C3': '7.4', 'C4': '8.8', 'C5': '8.5', 'C6': '8.7', 'C7': '7.7', 'C8': '8.7', 'C9': '8.9', 'C10': '8.5', 'C11': '7.2', 'C12': '7.7', 'C13': '7.1', 'C14': '7.6', 'C15': '8.4'}}, {'source': 'cost.csv', 'values': {'plant': 'F4', 'fixed_cost': '10290', 'capacity': '86', 'C1': '7', 'C2': '7.1', 'C3': '6.5', 'C4': '7.9', 'C5': '7.4', 'C6': '7.7', 'C7': '6.7', 'C8': '7.9', 'C9': '7.8', 'C10': '7.3', 'C11': '6.8', 'C12': '7', 'C13': '6.5', 'C14': '6.7', 'C15': '7.6'}}, {'source': 'cost.csv', 'values': {'plant': 'F5', 'fixed_cost': '16740', 'capacity': '157', 'C1': '3.5', 'C2': '3.8', 'C3': '2.9', 'C4': '4.3', 'C5': '3.6', 'C6': '3.9', 'C7': '3.2', 'C8': '4.3', 'C9': '4.5', 'C10': '4', 'C11': '3.2', 'C12': '4', 'C13': '2.9', 'C14': '3.4', 'C15': '3.9'}}, {'source': 'cost.csv', 'values': {'plant': 'F6', 'fixed_cost': '13960', 'capacity': '133', 'C1': '8.2', 'C2': '8.6', 'C3': '7.9', 'C4': '9.5', 'C5': '8.5', 'C6': '9.3', 'C7': '8.5', 'C8': '9.4', 'C9': '9', 'C10': '9.2', 'C11': '8.1', 'C12': '8.7', 'C13': '7.9', 'C14': '8.5', 'C15': '9'}}, {'source': 'cost.csv', 'values': {'plant': 'F7', 'fixed_cost': '12680', 'capacity': '118', 'C1': '6.9', 'C2': '7.6', 'C3': '6.8', 'C4': '8.4', 'C5': '8', 'C6': '8', 'C7': '7.6', 'C8': '8', 'C9': '8.1', 'C10': '7.8', 'C11': '6.9', 'C12': '7.1', 'C13': '7', 'C14': '6.9', 'C15': '7.5'}}, {'source': 'cost.csv', 'values': {'plant': 'F8', 'fixed_cost': '17890', 'capacity': '162', 'C1': '6.9', 'C2': '7.8', 'C3': '7.1', 'C4': '8.7', 'C5': '8.6', 'C6': '8.2', 'C7': '7.2', 'C8': '7.9', 'C9': '8.4', 'C10': '7.9', 'C11': '7', 'C12': '7.4', 'C13': '6.8', 'C14': '7.3', 'C15': '8'}}, {'source': 'cost.csv', 'values': {'plant': 'F9', 'fixed_cost': '10950', 'capacity': '92', 'C1': '3.5', 'C2': '3.8', 'C3': '2.8', 'C4': '4.4', 'C5': '4.2', 'C6': '4.8', 'C7': '3.8', 'C8': '5', 'C9': '4.5', 'C10': '4.1', 'C11': '3.2', 'C12': '3.7', 'C13': '3.7', 'C14': '3.2', 'C15': '4.5'}}, {'source': 'cost.csv', 'values': {'plant': 'F10', 'fixed_cost': '15320', 'capacity': '144', 'C1': '5.2', 'C2': '6.1', 'C3': '5.1', 'C4': '6.3', 'C5': '6.1', 'C6': '6', 'C7': '5.6', 'C8': '6.5', 'C9': '6.2', 'C10': '5.9', 'C11': '5.3', 'C12': '6.1', 'C13': '5.1', 'C14': '5.2', 'C15': '6.2'}}, {'source': 'cost.csv', 'values': {'plant': 'F11', 'fixed_cost': '11830', 'capacity': '107', 'C1': '5.2', 'C2': '5.5', 'C3': '4.5', 'C4': '6.2', 'C5': '5.7', 'C6': '6.1', 'C7': '5.1', 'C8': '5.8', 'C9': '5.7', 'C10': '6.2', 'C11': '5.2', 'C12': '5.2', 'C13': '4.5', 'C14': '5.1', 'C15': '5.4'}}, {'source': 'cost.csv', 'values': {'plant': 'F12', 'fixed_cost': '14110', 'capacity': '129', 'C1': '7.8', 'C2': '8.7', 'C3': '7.6', 'C4': '9', 'C5': '8.6', 'C6': '9', 'C7': '8.5', 'C8': '9.3', 'C9': '9.3', 'C10': '8.4', 'C11': '7.9', 'C12': '8.2', 'C13': '7.4', 'C14': '7.6', 'C15': '8.7'}}, {'source': 'cost.csv', 'values': {'plant': 'F13', 'fixed_cost': '15970', 'capacity': '151', 'C1': '6.7', 'C2': '6.6', 'C3': '6.1', 'C4': '7.3', 'C5': '7.1', 'C6': '7.5', 'C7': '6.7', 'C8': '8', 'C9': '7.6', 'C10': '7.2', 'C11': '6.3', 'C12': '6.9', 'C13': '6.2', 'C14': '6', 'C15': '7.2'}}, {'source': 'cost.csv', 'values': {'plant': 'F14', 'fixed_cost': '13140', 'capacity': '113', 'C1': '7.5', 'C2': '8.6', 'C3': '7.6', 'C4': '8.2', 'C5': '8', 'C6': '7.9', 'C7': '7.5', 'C8': '8.7', 'C9': '8.8', 'C10': '8.1', 'C11': '7.2', 'C12': '7.3', 'C13': '7', 'C14': '7', 'C15': '8'}}, {'source': 'cost.csv', 'values': {'plant': 'F15', 'fixed_cost': '10580', 'capacity': '85', 'C1': '5.1', 'C2': '5.8', 'C3': '4.6', 'C4': '5.9', 'C5': '6.5', 'C6': '5.9', 'C7': '5.2', 'C8': '7', 'C9': '7.1', 'C10': '5.9', 'C11': '5.1', 'C12': '5.8', 'C13': '5.4', 'C14': '4.9', 'C15': '6'}}, {'source': 'demand.csv', 'values': {'customer': 'C1', 'demand': '83'}}, {'source': 'demand.csv', 'values': {'customer': 'C2', 'demand': '76'}}, {'source': 'demand.csv', 'values': {'customer': 'C3', 'demand': '91'}}, {'source': 'demand.csv', 'values': {'customer': 'C4', 'demand': '68'}}, {'source': 'demand.csv', 'values': {'customer': 'C5', 'demand': '104'}}, {'source': 'demand.csv', 'values': {'customer': 'C6', 'demand': '97'}}, {'source': 'demand.csv', 'values': {'customer': 'C7', 'demand': '88'}}, {'source': 'demand.csv', 'values': {'customer': 'C8', 'demand': '73'}}, {'source': 'demand.csv', 'values': {'customer': 'C9', 'demand': '109'}}, {'source': 'demand.csv', 'values': {'customer': 'C10', 'demand': '95'}}, {'source': 'demand.csv', 'values': {'customer': 'C11', 'demand': '82'}}, {'source': 'demand.csv', 'values': {'customer': 'C12', 'demand': '67'}}, {'source': 'demand.csv', 'values': {'customer': 'C13', 'demand': '113'}}, {'source': 'demand.csv', 'values': {'customer': 'C14', 'demand': '79'}}, {'source': 'demand.csv', 'values': {'customer': 'C15', 'demand': '92'}}]
import gurobipy as gp
from gurobipy import GRB
plants = []
fixed_cost = {}
capacity = {}
customers = []
demand = {}
transport_cost = {}
for rec in LEGACY_RECORDS:
    vals = rec['values']
    if rec['source'] == 'cost.csv':
        plant = vals['plant']
        plants.append(plant)
        fixed_cost[plant] = float(vals['fixed_cost'])
        capacity[plant] = float(vals['capacity'])
        transport_cost[plant] = {}
        for k, v in vals.items():
            if k.startswith('C'):
                transport_cost[plant][k] = float(v)
    elif rec['source'] == 'demand.csv':
        customer = vals['customer']
        customers.append(customer)
        demand[customer] = float(vals['demand'])
for plant in plants:
    if plant not in fixed_cost or plant not in capacity or plant not in transport_cost:
        raise ValueError(f'Missing plant data for {plant}')
    for customer in customers:
        if customer not in transport_cost[plant]:
            raise ValueError(f'Missing transport cost for {plant}, {customer}')
for customer in customers:
    if customer not in demand:
        raise ValueError(f'Missing demand for {customer}')
m = gp.Model('Plant_Location')
y = m.addVars(plants, vtype=GRB.BINARY, name='')
x = m.addVars(plants, customers, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((fixed_cost[i] * y[i] for i in plants)) + gp.quicksum((transport_cost[i][j] * x[i, j] for i in plants for j in customers)), GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for i in plants)) == demand[j] for j in customers), name='')
m.addConstrs((gp.quicksum((x[i, j] for j in customers)) <= capacity[i] * y[i] for i in plants), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')