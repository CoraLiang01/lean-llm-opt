import gurobipy as gp
from gurobipy import GRB
options = ['OPT1', 'OPT2', 'OPT3', 'OPT4', 'OPT5', 'OPT6', 'OPT7', 'OPT8', 'OPT9', 'OPT10', 'OPT11', 'OPT12', 'OPT13', 'OPT14', 'OPT15', 'OPT16', 'OPT17', 'OPT18', 'OPT19', 'OPT20', 'OPT21', 'OPT22', 'OPT23', 'OPT24', 'OPT25', 'OPT26', 'OPT27', 'OPT28', 'OPT29', 'OPT30', 'OPT31', 'OPT32', 'OPT33', 'OPT34', 'OPT35', 'OPT36', 'OPT37', 'OPT38', 'OPT39', 'OPT40', 'OPT41', 'OPT42', 'OPT43', 'OPT44', 'OPT45', 'OPT46', 'OPT47', 'OPT48', 'OPT49', 'OPT50', 'OPT51', 'OPT52', 'OPT53', 'OPT54', 'OPT55', 'OPT56', 'OPT57', 'OPT58', 'OPT59', 'OPT60', 'OPT61', 'OPT62', 'OPT63', 'OPT64', 'OPT65', 'OPT66', 'OPT67', 'OPT68', 'OPT69', 'OPT70', 'OPT71', 'OPT72', 'OPT73', 'OPT74', 'OPT75', 'OPT76', 'OPT77', 'OPT78', 'OPT79', 'OPT80', 'OPT81', 'OPT82', 'OPT83', 'OPT84', 'OPT85', 'OPT86', 'OPT87', 'OPT88', 'OPT89', 'OPT90', 'OPT91', 'OPT92', 'OPT93', 'OPT94', 'OPT95', 'OPT96', 'OPT97', 'OPT98', 'OPT99', 'OPT100', 'OPT101', 'OPT102', 'OPT103', 'OPT104', 'OPT105', 'OPT106', 'OPT107', 'OPT108', 'OPT109', 'OPT110', 'OPT111', 'OPT112', 'OPT113', 'OPT114', 'OPT115', 'OPT116', 'OPT117', 'OPT118', 'OPT119', 'OPT120']
assets = ['AST1', 'AST2', 'AST3', 'AST4', 'AST5', 'AST6']
Cost = {'OPT1': 1.12, 'OPT2': 0.98, 'OPT3': 1.05, 'OPT4': 1.2, 'OPT5': 1.15, 'OPT6': 1.1, 'OPT7': 1.08, 'OPT8': 1.13, 'OPT9': 1.09, 'OPT10': 1.07, 'OPT11': 1.14, 'OPT12': 1.06, 'OPT13': 1.11, 'OPT14': 1.16, 'OPT15': 1.17, 'OPT16': 1.18, 'OPT17': 1.19, 'OPT18': 1.21, 'OPT19': 1.22, 'OPT20': 1.23, 'OPT21': 1.24, 'OPT22': 1.25, 'OPT23': 1.26, 'OPT24': 1.27, 'OPT25': 1.28, 'OPT26': 1.29, 'OPT27': 1.3, 'OPT28': 1.31, 'OPT29': 1.32, 'OPT30': 1.33, 'OPT31': 1.34, 'OPT32': 1.35, 'OPT33': 1.36, 'OPT34': 1.37, 'OPT35': 1.38, 'OPT36': 1.39, 'OPT37': 1.4, 'OPT38': 1.41, 'OPT39': 1.42, 'OPT40': 1.43, 'OPT41': 1.44, 'OPT42': 1.45, 'OPT43': 1.46, 'OPT44': 1.47, 'OPT45': 1.48, 'OPT46': 1.49, 'OPT47': 1.5, 'OPT48': 1.51, 'OPT49': 1.52, 'OPT50': 1.53, 'OPT51': 1.54, 'OPT52': 1.55, 'OPT53': 1.56, 'OPT54': 1.57, 'OPT55': 1.58, 'OPT56': 1.59, 'OPT57': 1.6, 'OPT58': 1.61, 'OPT59': 1.62, 'OPT60': 1.63, 'OPT61': 1.64, 'OPT62': 1.65, 'OPT63': 1.66, 'OPT64': 1.67, 'OPT65': 1.68, 'OPT66': 1.69, 'OPT67': 1.7, 'OPT68': 1.71, 'OPT69': 1.72, 'OPT70': 1.73, 'OPT71': 1.74, 'OPT72': 1.75, 'OPT73': 1.76, 'OPT74': 1.77, 'OPT75': 1.78, 'OPT76': 1.79, 'OPT77': 1.8, 'OPT78': 1.81, 'OPT79': 1.82, 'OPT80': 1.83, 'OPT81': 1.84, 'OPT82': 1.85, 'OPT83': 1.86, 'OPT84': 1.87, 'OPT85': 1.88, 'OPT86': 1.89, 'OPT87': 1.9, 'OPT88': 1.91, 'OPT89': 1.92, 'OPT90': 1.93, 'OPT91': 1.94, 'OPT92': 1.95, 'OPT93': 1.96, 'OPT94': 1.97, 'OPT95': 1.98, 'OPT96': 1.99, 'OPT97': 2.0, 'OPT98': 2.01, 'OPT99': 2.02, 'OPT100': 2.03, 'OPT101': 2.04, 'OPT102': 2.05, 'OPT103': 2.06, 'OPT104': 2.07, 'OPT105': 2.08, 'OPT106': 2.09, 'OPT107': 2.1, 'OPT108': 2.11, 'OPT109': 2.12, 'OPT110': 2.13, 'OPT111': 2.14, 'OPT112': 2.15, 'OPT113': 2.16, 'OPT114': 2.17, 'OPT115': 2.18, 'OPT116': 2.19, 'OPT117': 2.2, 'OPT118': 2.21, 'OPT119': 2.22, 'OPT120': 2.23}
Delta = {opt: 0.01 * (i % 10 - 5) for i, opt in enumerate(options, 1)}
Gamma = {opt: 0.002 * (i % 7 - 3) for i, opt in enumerate(options, 1)}
Vega = {opt: 0.005 * (i % 8 - 4) for i, opt in enumerate(options, 1)}
MaxLong = {opt: 10 + i % 5 for i, opt in enumerate(options, 1)}
MaxShort = {opt: -10 - i % 5 for i, opt in enumerate(options, 1)}
A = {}
for i, opt in enumerate(options):
    A[opt] = {}
    for j, ast in enumerate(assets):
        if i // 20 == j:
            A[opt][ast] = 1
        elif i % 6 == j:
            A[opt][ast] = 1
        else:
            A[opt][ast] = 0
for opt in options:
    if opt not in Cost or opt not in Delta or opt not in Gamma or (opt not in Vega) or (opt not in MaxLong) or (opt not in MaxShort) or (opt not in A):
        raise ValueError(f'Missing data for option {opt}')
    if set(A[opt].keys()) != set(assets):
        raise ValueError(f'Asset reference matrix incomplete for option {opt}')
G_initial = {'Delta': 0.25, 'Gamma': 0.08, 'Vega': 0.17}
Tolerance = {'Delta': 0.06, 'Gamma': 0.05, 'Vega': 0.07}
m = gp.Model('Option_Hedging')
x = m.addVars(options, vtype=GRB.INTEGER, lb={opt: MaxShort[opt] for opt in options}, ub={opt: MaxLong[opt] for opt in options}, name='')
absx = m.addVars(options, vtype=GRB.INTEGER, lb=0, name='')
for opt in options:
    m.addConstr(absx[opt] >= x[opt], name=f'absx_pos_{opt}')
    m.addConstr(absx[opt] >= -x[opt], name=f'absx_neg_{opt}')
m.setObjective(gp.quicksum((Cost[opt] * absx[opt] for opt in options)), GRB.MINIMIZE)
for greek, Gvec, Ginit, tol in [('Delta', Delta, G_initial['Delta'], Tolerance['Delta']), ('Gamma', Gamma, G_initial['Gamma'], Tolerance['Gamma']), ('Vega', Vega, G_initial['Vega'], Tolerance['Vega'])]:
    expr = Ginit + gp.quicksum((Gvec[opt] * A[opt][ast] * x[opt] for opt in options for ast in assets))
    m.addConstr(expr <= tol, name=f'{greek}_upper')
    m.addConstr(expr >= -tol, name=f'{greek}_lower')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')