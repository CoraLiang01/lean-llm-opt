import gurobipy as gp
from gurobipy import GRB
devices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
device_capacities = {'A': 3500, 'B': 4200, 'C': 4500, 'D': 2800, 'E': 3300, 'F': 3800, 'G': 4100, 'H': 3900, 'I': 4800, 'J': 3100}
products = [f'P{k}' for k in range(1, 112)]
unit_profits = {'P1': 12.5, 'P2': 15.0, 'P3': 13.2, 'P4': 14.8, 'P5': 16.1, 'P6': 11.9, 'P7': 17.3, 'P8': 12.7, 'P9': 13.5, 'P10': 14.2, 'P11': 15.6, 'P12': 13.9, 'P13': 16.4, 'P14': 12.3, 'P15': 14.5, 'P16': 15.2, 'P17': 13.7, 'P18': 16.0, 'P19': 12.8, 'P20': 14.9, 'P21': 15.3, 'P22': 13.1, 'P23': 16.2, 'P24': 12.6, 'P25': 14.1, 'P26': 15.7, 'P27': 13.8, 'P28': 16.5, 'P29': 12.4, 'P30': 14.6, 'P31': 15.1, 'P32': 13.6, 'P33': 16.3, 'P34': 12.9, 'P35': 14.7, 'P36': 15.4, 'P37': 13.3, 'P38': 16.6, 'P39': 12.2, 'P40': 14.3, 'P41': 15.8, 'P42': 13.4, 'P43': 16.7, 'P44': 12.1, 'P45': 14.4, 'P46': 15.5, 'P47': 13.0, 'P48': 16.8, 'P49': 12.0, 'P50': 14.0, 'P51': 15.9, 'P52': 13.5, 'P53': 16.9, 'P54': 12.7, 'P55': 14.8, 'P56': 15.6, 'P57': 13.2, 'P58': 16.1, 'P59': 12.3, 'P60': 14.5, 'P61': 15.0, 'P62': 13.9, 'P63': 16.4, 'P64': 12.8, 'P65': 14.9, 'P66': 15.3, 'P67': 13.7, 'P68': 16.2, 'P69': 12.6, 'P70': 14.1, 'P71': 15.7, 'P72': 13.8, 'P73': 16.5, 'P74': 12.4, 'P75': 14.6, 'P76': 15.1, 'P77': 13.6, 'P78': 16.3, 'P79': 12.9, 'P80': 14.7, 'P81': 15.4, 'P82': 13.3, 'P83': 16.6, 'P84': 12.2, 'P85': 14.3, 'P86': 15.8, 'P87': 13.4, 'P88': 16.7, 'P89': 12.1, 'P90': 14.4, 'P91': 15.5, 'P92': 13.0, 'P93': 16.8, 'P94': 12.0, 'P95': 14.0, 'P96': 15.9, 'P97': 13.5, 'P98': 16.9, 'P99': 12.7, 'P100': 14.8, 'P101': 15.6, 'P102': 13.2, 'P103': 16.1, 'P104': 12.3, 'P105': 14.5, 'P106': 15.0, 'P107': 13.9, 'P108': 16.4, 'P109': 12.8, 'P110': 14.9, 'P111': 15.3}
device_times = {'A': {p: 1.1 + 0.01 * i for (i, p) in enumerate(products)}, 'B': {p: 1.2 + 0.01 * i for (i, p) in enumerate(products)}, 'C': {p: 1.3 + 0.01 * i for (i, p) in enumerate(products)}, 'D': {p: 1.4 + 0.01 * i for (i, p) in enumerate(products)}, 'E': {p: 1.5 + 0.01 * i for (i, p) in enumerate(products)}, 'F': {p: 1.6 + 0.01 * i for (i, p) in enumerate(products)}, 'G': {p: 1.7 + 0.01 * i for (i, p) in enumerate(products)}, 'H': {p: 1.8 + 0.01 * i for (i, p) in enumerate(products)}, 'I': {p: 1.9 + 0.01 * i for (i, p) in enumerate(products)}, 'J': {p: 2.0 + 0.01 * i for (i, p) in enumerate(products)}}
for p in products:
    if p not in unit_profits:
        raise ValueError(f'Missing unit profit for product {p}')
    for d in devices:
        if p not in device_times[d]:
            raise ValueError(f'Missing device time for device {d}, product {p}')
for d in devices:
    if d not in device_capacities:
        raise ValueError(f'Missing capacity for device {d}')
m = gp.Model('factory_production')
x_vars = m.addVars(products, lb=0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((unit_profits[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((device_times[d][p] * x_vars[p] for p in products)) <= device_capacities[d] for d in devices), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')