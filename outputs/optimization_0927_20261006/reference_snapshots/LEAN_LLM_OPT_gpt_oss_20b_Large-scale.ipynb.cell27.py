ORIGINAL_CODE_PROMPT = """
    You are an expert in mathematical optimization and Python programming. Your task is to write Python code to solve the provided mathematical optimization model using the Gurobi library. The code should include the definition of the objective function, constraints, and decision variables. Please don't add additional explanations. Please don't include ```python and ```.Below is the provided mathematical optimization model:

    Mathematical Optimization Model:
    {output}
    """

QUERY_ONLY_CODE_EXAMPLES = {
    "NRM": """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\max \\quad \\sum_i A_i \\cdot x_i$
Constraints
1. Inventory Constraints:
$\\quad \\quad x_i \\leq I_i, \\quad \forall i$
2. Demand Constraints:
$x_i \\leq d_i, \\quad \forall i$
3. Startup Constraint:
$\\sum_i x_i \\geq s$
Retrieved Information
$\\small I = [7550, 6244]$
$\\small A = [149, 389]$
$\\small d = [15057, 12474]$
$\\small s = 100$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Product_Optimization")

# Decision variables for the number of units of each product
x_1 = m.addVar(vtype=GRB.INTEGER, name="x_1") # Number of units of product 1
x_2 = m.addVar(vtype=GRB.INTEGER, name="x_2") # Number of units of product 2

# Objective function: Maximize 149 x_1 + 389 x_2
m.setObjective(149 * x_1 + 389 * x_2, GRB.MAXIMIZE)

# Constraints
m.addConstr(x_1 <= 7550, name="inventory_constraint_1")
m.addConstr(x_2 <= 6244, name="inventory_constraint_2")
m.addConstr(x_1 <= 15057, name="demand_constraint_1")
m.addConstr(x_2 <= 12474, name="demand_constraint_2")

# Non-negativity constraints are implicitly handled by the integer constraints (x_1, x_2 >= 0)

# Solve the model
m.optimize()

        """,
    "FLP": """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\min \\quad \\sum_{i} \\sum_{j} A_{ij} \\cdot x_{ij} + \\sum_{i} c_i \\cdot y_i$

Constraints
1. Demand Constraint:
$\\quad \\quad \\sum_i x_{ij} = d_j, \\quad \forall j$
2. Capacity Constraint:
$\\quad \\quad \\sum_j x_{ij} \\leq M \\cdot y_i, \\quad \forall i$
3. Non-negativity:
$\\quad \\quad x_{ij} \\geq 0, \\quad \forall i,j$
4. Binary Requirement:
$\\quad \\quad y_i \\in \\{0,1\\}, \\quad \forall i$

Retrieved Information
$\\small d = [1083, 776, 16214, 553, 17106, 594, 732]$
$\\small c = [102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16]$
$\\small A = \begin{bmatrix}
1506.22 & 70.90 & 8.44 & 260.27 & 197.47 & 71.71 & 61.19 \\  
1732.65 & 1780.72 & 567.44 & 448.68 & 29.00 & 1484.91 & 963.92 \\  
115.66 & 100.76 & 64.68 & 1324.53 & 64.99 & 134.88 & 2102.83 \\  
1254.78 & 1115.63 & 52.31 & 1036.16 & 892.63 & 1464.04 & 1383.41 \\  
42.90 & 891.01 & 1013.94 & 1128.72 & 58.91 & 42.89 & 1570.31 \\  
0.70 & 139.46 & 70.03 & 79.15 & 1482.00 & 0.91 & 110.46 \\  
1732.30 & 1780.44 & 486.50 & 523.74 & 522.08 & 82.48 & 826.41
\\end{bmatrix}$
$\\small M = \\sum_j d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 38058 $

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
d = np.array([1083, 776, 16214, 553, 17106, 594, 732])
c = np.array([102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16])
A = np.array([[1506.22, 70.90, 8.44, 260.27, 197.47, 71.71, 61.19],  
[1732.65, 1780.72, 567.44, 448.68, 29.00, 1484.91, 963.92],  
[115.66, 100.76, 64.68, 1324.53, 64.99, 134.88, 2102.83],  
[1254.78, 1115.63, 52.31, 1036.16, 892.63, 1464.04, 1383.41],  
[42.90, 891.01, 1013.94, 1128.72, 58.91, 42.89, 1570.31],  
[0.70, 139.46, 70.03, 79.15, 1482.00, 0.91, 110.46],  
[1732.30, 1780.44, 486.50, 523.74, 522.08, 82.48, 826.41]])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(A.shape[0], A.shape[1], lb=0, name="x")
y = m.addVars(A.shape[0], vtype=GRB.BINARY, name="y")

# Objective function
m.setObjective(gp.quicksum(A[i, j]*x[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])) + gp.quicksum(c[i]*y[i] for i in range(A.shape[0])), GRB.MINIMIZE)

# Constraints
for j in range(A.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(A.shape[0])) == d[j], name=f"demand_constraint_{j}")

M = 1000000  # large number
for i in range(A.shape[0]):
    m.addConstr(-M*y[i] + gp.quicksum(x[i, j] for j in range(A.shape[1])) <= 0, name=f"M_constraint_{i}")

# Solve the model
m.optimize()
        """,
    "AP": """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\min \\quad \\sum_{i=1}^3 \\sum_{j=1}^3 c_{ij} \\cdot x_{ij}$

Constraints
1. Row Assignment Constraint:
$\\quad \\quad \\sum_{j=1}^3 x_{ij} = 1, \\quad \forall i \\in \\{1,2,3\\}$
2. Column Assignment Constraint:
$\\quad \\quad \\sum_{i=1}^3 x_{ij} = 1, \\quad \forall j \\in \\{1,2,3\\}$
3. Binary Constraint:
$\\quad \\quad x_{ij} \\in \\{0,1\\}, \\quad \forall i,j$

Retrieved Information
$\\small c = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 
\\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [3000, 3200, 3100],
    [2800, 3300, 2900],
    [2900, 3100, 3000]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize()
""",
    "TP": """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\min \\quad \\sum_i \\sum_j c_{ij} \\cdot x_{ij}$

Constraints
1. Demand Constraint:
$\\quad \\quad \\sum_i x_{ij} \\geq d_j, \\quad \forall j$
2. Capacity Constraint:
$\\quad \\quad \\sum_j x_{ij} \\leq s_i, \\quad \forall i$

Retrieved Information
$\\small d = [94, 39, 65, 435]$
$\\small s = [2531, 20, 210, 241]$
$\\small c = \begin{bmatrix}
883.91 & 0.04 & 0.03 & 44.45 \\
543.75 & 23.68 & 23.67 & 447.75 \\
537.34 & 23.76 & 498.95 & 440.60 \\
1791.49 & 68.21 & 1432.48 & 1527.76
\\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization")

# Decision variables
x_S1_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C1")
x_S1_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C2")
x_S1_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C3")
x_S1_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S1_C4")
x_S2_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C1")
x_S2_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C2")
x_S2_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C3")
x_S2_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S2_C4")
x_S3_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C1")
x_S3_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C2")
x_S3_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C3")
x_S3_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S3_C4")
x_S4_C1 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C1")
x_S4_C2 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C2")
x_S4_C3 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C3")
x_S4_C4 = m.addVar(vtype=GRB.INTEGER, name="x_S4_C4")

# Objective function
m.setObjective(883.91 * x_S2_C1 + 0.04 * x_S2_C2 + 0.03 * x_S2_C3 + 44.45 * x_S2_C4 + 543.75 * x_S1_C1 + 23.68 * x_S1_C2 + 23.67 * x_S1_C3 + 447.75 * x_S1_C4 + 537.34 * x_S3_C1 + 23.76 * x_S3_C2 + 498.95 * x_S3_C3 + 440.60 * x_S3_C4 + 1791.49 * x_S4_C1 + 68.21 * x_S4_C2 + 1432.48 * x_S4_C3 + 1527.76 * x_S4_C4, GRB.MINIMIZE)

# Constraints
m.addConstr(x_S1_C1 + x_S2_C1 + x_S3_C1 + x_S4_C1 >= 94, name="demand_constraint1")
m.addConstr(x_S1_C2 + x_S2_C2 + x_S3_C2 + x_S4_C2 >= 39, name="demand_constraint2")
m.addConstr(x_S1_C3 + x_S2_C3 + x_S3_C3 + x_S4_C3 >= 65, name="demand_constraint3")
m.addConstr(x_S1_C4 + x_S2_C4 + x_S3_C4 + x_S4_C4 >= 435, name="demand_constraint4")
m.addConstr(x_S1_C1 + x_S1_C2 + x_S1_C3 + x_S1_C4 <= 2531, name="capacity_constraint1")
m.addConstr(x_S2_C1 + x_S2_C2 + x_S2_C3 + x_S2_C4 <= 20, name="capacity_constraint2")
m.addConstr(x_S3_C1 + x_S3_C2 + x_S3_C3 + x_S3_C4 <= 210, name="capacity_constraint3")
m.addConstr(x_S4_C1 + x_S4_C2 + x_S4_C3 + x_S4_C4 <= 241, name="capacity_constraint4")

# Solve the model
m.optimize()
        """,
    "RA": """
For example, here is a simple instance for reference:

Always remember: If not specified. All the variables are non-negative interger.

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\max \\quad \\sum_i \\sum_j p_i \\cdot x_{ij}$

Constraints
1. Capacity Constraint:
$\\quad \\quad \\sum_i a_i \\cdot x_{ij} \\leq c_j, \\quad \forall j$
2. Non-negativity Constraint:
$\\quad \\quad x_{ij} \\geq 0, \\quad \forall i,j$

Retrieved Information
$\\small p = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]$
$\\small a = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]$
$\\small c = [4466]$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(20, vtype=GRB.INTEGER, name="x")

# Objective function
m.setObjective(sum(x[i]*c[i] for i in range(20)), GRB.MAXIMIZE)

# Constraints
m.addConstr(sum(x[i]*w[i] for i in range(20)) <= 4466, name="capacity_constraint")

# Coefficients for the objective function
c = [321, 309, 767, 300, 763, 318, 871, 522, 300, 275, 858, 593, 126, 460, 685, 443, 700, 522, 940, 598]

# Coefficients for the capacity constraint
w = [495, 123, 165, 483, 472, 258, 425, 368, 105, 305, 482, 387, 469, 341, 318, 104, 377, 213, 56, 131]

# Solve the model
m.optimize()
```

-----
Here is another simple instance for reference:

Objective Function:
$\\quad \\quad \\max \\quad \\sum_i p_i \\cdot x_i$

Constraints
1. Capacity Constraint:
$\\quad \\quad \\sum_i a_i \\cdot x_i \\leq 180$
2. Dependency Constraint:
$\\quad \\quad x_1 \\leq x_3$
3. Non-negativity Constraint:
$\\quad \\quad x_i \\geq 0, \\quad \forall i$

Retrieved Information
$\\small p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]$
$\\small a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]$

The corresponding Python code for this instance is as follows:

import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(10, vtype=GRB.INTEGER, name="x")

# Objective function
p = [888, 134, 129, 370, 921, 765, 154, 837, 584, 365]
m.setObjective(sum(x[i]*p[i] for i in range(10)), GRB.MAXIMIZE)

# Constraints
a = [4, 2, 4, 3, 2, 1, 2, 1, 3, 3]
m.addConstr(sum(x[i]*a[i] for i in range(10)) <= 180, name="capacity_constraint")
m.addConstr(x[0] <= x[2], name="dependency_constraint")

# Solve the model
m.optimize()
        
        """,
    "Others": """
For example, here is a simple instance for reference:

Mathematical Optimization Model:
Maximize 5x_S + 8x_F
Subject to
    2x_S + 5x_F <= 200
    x_S <= 0.3(x_S + x_F)
    x_F >= 10
    x_S, x_F _ Z+

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Create the model
m = gp.Model("Worker_Optimization")

# Decision variables for the number of seasonal (x_S) and full-time (x_F) workers
x_S = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_S")  # Number of seasonal workers
x_F = m.addVar(vtype=GRB.INTEGER, lb=0, name="x_F")  # Number of full-time workers

# Objective function: Maximize Z = 5x_S + 8x_F
m.setObjective(5 * x_S + 8 * x_F, GRB.MAXIMIZE)

# Constraints
m.addConstr(2 * x_S + 5 * x_F <= 200, name="resource_constraint")
m.addConstr(x_S <= 0.3 * (x_S + x_F), name="seasonal_ratio_constraint")
m.addConstr(x_F >= 10, name="full_time_minimum_constraint")

# Non-negativity constraints are implicitly handled by the integer constraints (x_S, x_F >= 0)

# Solve the model
m.optimize()
```
The another example is:

Mathematical Optimization Model:
Minimize 919x_11 + 556x_12 + 951x_13 + 21x_21 + 640x_22 + 409x_23 + 59x_31 + 786x_32 + 304x_33
Subject to
    x_11 + x_12 + x_13 = 1
    x_21 + x_22 + x_23 = 1
    x_31 + x_32 + x_33 = 1
    x_11 + x_21 + x_31 = 1
    x_12 + x_22 + x_32 = 1
    x_13 + x_23 + x_33 = 1
    x_11, x_12, x_13, x_21, x_22, x_23, x_31, x_32, x_33 ∈ {{0,1}}

The corresponding Python code for this instance is as follows:

```python

import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [919, 556, 951],
    [21, 640, 409],
    [59, 786, 304]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize() 
```


-----  Here is a Capacitated Facility Location (MIP) instance for reference:----- 

Mathematical Optimization Model:
\\[
\begin{{aligned}}
\\min\\;& 10000\\,y_1 + 15010\\,y_2 + 12000\\,y_3 \\
& + 5x_{{11}} + 7x_{{12}} + 3x_{{13}} + 4x_{{14}} \\
& + 6x_{{21}} + 4x_{{22}} + 5x_{{23}} + 3x_{{24}} \\
& + 2x_{{31}} + 6x_{{32}} + 7x_{{33}} + 4x_{{34}} \\
\text{s.t.}\\;& \\sum_{i=1}^{3} x_{ij} \\ge d_j && \forall j=1,\\dots,4 \\
& \\sum_{j=1}^{4} x_{ij} \\le s_i\\,y_i && \forall i=1,\\dots,3 \\
& d = \\{100, 150, 200, 120\\} \\
& s = \\{300, 400, 250\\} \\
& y_i \\in \\{0,1\\} && \forall i \\
& x_{ij} \\ge 0 && \forall i,j
\\end{{aligned}}
\\]

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

# Data
facilities = [1, 2, 3]
customers = [1, 2, 3, 4]

# Fixed costs
fixed_costs = {1: 10000, 2: 15010, 3: 12000}

# Capacities
capacities = {1: 300, 2: 400, 3: 250}

# Demands
demands = {1: 100, 2: 150, 3: 200, 4: 120}

# Variable costs
var_costs = {
    (1, 1): 5, (1, 2): 7, (1, 3): 3, (1, 4): 4,
    (2, 1): 6, (2, 2): 4, (2, 3): 5, (2, 4): 3,
    (3, 1): 2, (3, 2): 6, (3, 3): 7, (3, 4): 4,
}
arcs = var_costs.keys()

m = gp.Model("CFLP_Example")

# Variables
y = m.addVars(facilities, vtype=GRB.BINARY, name="y")
x = m.addVars(arcs, name="x") # default lb=0

# Objective
obj_fixed = y.prod(fixed_costs)
obj_variable = x.prod(var_costs)
m.setObjective(obj_fixed + obj_variable, GRB.MINIMIZE)

# Demand Constraints
for j in customers:
    m.addConstr(x.sum('*', j) >= demands[j], name=f"demand_{j}")

# Capacity Constraints
for i in facilities:
    m.addConstr(x.sum(i, '*') <= capacities[i] * y[i], name=f"capacity_{i}")
    
m.optimize()
```

----- Here is a Traveling Salesperson Problem (TSP) instance for reference: -----

Mathematical Optimization Model:
\\[
\begin{aligned}
\\min \\quad & 15x_{12}+25x_{13}+35x_{14}+18x_{21}+30x_{23}+40x_{24} \\
& + 28x_{31}+20x_{32}+38x_{34}+45x_{41}+50x_{42}+55x_{43} \\
\text{s.t.}\\quad & \\sum_{j
eq i} x_{ij}=1 && \forall i=1,\\dots,4 \\
& \\sum_{i
eq j} x_{ij}=1 && \forall j=1,\\dots,4 \\
& u_i - u_j + 4x_{ij} \\le 3 && \forall i,j \\in \\{1,\\dots,4\\}, i 
eq j \\
& 1 \\le u_i \\le 4 && \forall i=1,\\dots,4 \\
& x_{ij} \\in \\{0,1\\} && \forall i 
eq j \\
& u_i \\in \\mathbb{Z} && \forall i
\\end{aligned}
\\]

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

n = 4
nodes = range(1, n + 1)

costs = {
    (1, 2): 15, (1, 3): 25, (1, 4): 35,
    (2, 1): 18, (2, 3): 30, (2, 4): 40,
    (3, 1): 28, (3, 2): 20, (3, 4): 38,
    (4, 1): 45, (4, 2): 50, (4, 3): 55,
}
arcs = costs.keys()

m = gp.Model("TSP_Example")

# Variables
x = m.addVars(arcs, vtype=GRB.BINARY, name="x")
u = m.addVars(nodes, vtype=GRB.INTEGER, lb=1, ub=n, name="u")

# Objective
m.setObjective(x.prod(costs), GRB.MINIMIZE)

# Constraints
for i in nodes:
    m.addConstr(x.sum(i, '*') == 1, name=f"leave_{i}")

for j in nodes:
    m.addConstr(x.sum('*', j) == 1, name=f"enter_{j}")

# MTZ Subtour Elimination
for i, j in arcs:
    m.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, name=f"MTZ_{i}_{j}")

m.optimize()
```
----- Here is a Maximum Flow Problem (LP) instance for reference: -----

Mathematical Optimization Model:
\\[
\begin{aligned}
\\max\\;& F \\
\text{s.t. } 
& f_{01}+f_{02} = F && \text{(Source 0)} \\
& f_{12}+f_{13}+f_{14} = f_{01} && \text{(Node 1)} \\
& f_{23}+f_{24} = f_{02} + f_{12} && \text{(Node 2)} \\
& f_{34} = f_{13} + f_{23} && \text{(Node 3)} \\
& f_{14}+f_{24}+f_{34} = F && \text{(Sink 4)} \\
& f_{01} \\le 25, f_{02} \\le 35 \\
& f_{12} \\le 12, f_{13} \\le 18, f_{14} \\le 8 \\
& f_{23} \\le 12, f_{24} \\le 22 \\
& f_{34} \\le 30 \\
& f_{ij} \\ge 0, F \\ge 0
\\end{aligned}
\\]

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB

capacities = {
    (0, 1): 25, (0, 2): 35,
    (1, 2): 12, (1, 3): 18, (1, 4): 8,
    (2, 3): 12, (2, 4): 22,
    (3, 4): 30
}
arcs = capacities.keys()
nodes = [0, 1, 2, 3, 4]
source = 0
sink = 4

m = gp.Model("MaxFlow_Example")

# Variables
f = m.addVars(arcs, name="f") # default lb=0
F = m.addVar(name="F", lb=0)

# Objective
m.setObjective(F, GRB.MAXIMIZE)

# Capacity Constraints
m.addConstrs((f[i, j] <= capacities[i, j] for i, j in arcs), name="cap")

# Balance Constraints
# Source
m.addConstr(f.sum(source, '*') - f.sum('*', source) == F, name="source_balance")

# Sink
m.addConstr(f.sum('*', sink) - f.sum(sink, '*') == F, name="sink_balance")

# Transshipment nodes
for i in [1, 2, 3]:
    m.addConstr(f.sum('*', i) - f.sum(i, '*') == 0, name=f"balance_{i}")

m.optimize()
```
""",
}
QUERY_ONLY_CODE_ALIASES = {
    "Network Revenue Management": "NRM",
    "Network Revenue Management Problem": "NRM",
    "Facility Location Problem": "FLP",
    "Facility Location": "FLP",
    "Assignment Problem": "AP",
    "Assignment": "AP",
    "Transportation Problem": "TP",
    "Transportation": "TP",
    "Resource Allocation": "RA",
    "Resource Allocation Problem": "RA"
}

CSV_CODE_GUIDANCE = {
    'FLP': """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\min \\quad \\sum_{i} \\sum_{j} A_{ij} \\cdot x_{ij} + \\sum_{i} c_i \\cdot y_i$

Constraints
1. Demand Constraint:
$\\quad \\quad \\sum_i x_{ij} = d_j, \\quad \\forall j$
2. Capacity Constraint:
$\\quad \\quad \\sum_j x_{ij} \\leq M \\cdot y_i, \\quad \\forall i$
3. Non-negativity:
$\\quad \\quad x_{ij} \\geq 0, \\quad \\forall i,j$
4. Binary Requirement:
$\\quad \\quad y_i \\in \\{0,1\\}, \\quad \\forall i$

Retrieved Information
$\\small d = [1083, 776, 16214, 553, 17106, 594, 732]$
$\\small c = [102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16]$
$\\small A = \\begin{bmatrix}
1506.22 & 70.90 & 8.44 & 260.27 & 197.47 & 71.71 & 61.19 \\\\  
1732.65 & 1780.72 & 567.44 & 448.68 & 29.00 & 1484.91 & 963.92 \\\\  
115.66 & 100.76 & 64.68 & 1324.53 & 64.99 & 134.88 & 2102.83 \\\\  
1254.78 & 1115.63 & 52.31 & 1036.16 & 892.63 & 1464.04 & 1383.41 \\\\  
42.90 & 891.01 & 1013.94 & 1128.72 & 58.91 & 42.89 & 1570.31 \\\\  
0.70 & 139.46 & 70.03 & 79.15 & 1482.00 & 0.91 & 110.46 \\\\  
1732.30 & 1780.44 & 486.50 & 523.74 & 522.08 & 82.48 & 826.41
\\end{bmatrix}$
$\\small M = \\sum_j d_j = 1083 + 776 + 16214 + 553 + 17106 + 594 + 732 = 38058 $

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
d = np.array([1083, 776, 16214, 553, 17106, 594, 732])
c = np.array([102.33, 94.92, 91.83, 98.71, 95.73, 99.96, 98.16])
A = np.array([[1506.22, 70.90, 8.44, 260.27, 197.47, 71.71, 61.19],  
[1732.65, 1780.72, 567.44, 448.68, 29.00, 1484.91, 963.92],  
[115.66, 100.76, 64.68, 1324.53, 64.99, 134.88, 2102.83],  
[1254.78, 1115.63, 52.31, 1036.16, 892.63, 1464.04, 1383.41],  
[42.90, 891.01, 1013.94, 1128.72, 58.91, 42.89, 1570.31],  
[0.70, 139.46, 70.03, 79.15, 1482.00, 0.91, 110.46],  
[1732.30, 1780.44, 486.50, 523.74, 522.08, 82.48, 826.41]])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(A.shape[0], A.shape[1], lb=0, name="x")
y = m.addVars(A.shape[0], vtype=GRB.BINARY, name="y")

# Objective function
m.setObjective(gp.quicksum(A[i, j]*x[i, j] for i in range(A.shape[0]) for j in range(A.shape[1])) + gp.quicksum(c[i]*y[i] for i in range(A.shape[0])), GRB.MINIMIZE)

# Constraints
for j in range(A.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(A.shape[0])) == d[j], name=f"demand_constraint_{j}")

M = 1000000  # large number
for i in range(A.shape[0]):
    m.addConstr(-M*y[i] + gp.quicksum(x[i, j] for j in range(A.shape[1])) <= 0, name=f"M_constraint_{i}")

# Solve the model
m.optimize()
        """,
    'AP': """
For example, here is a simple instance for reference:

Mathematical Optimization Model:

Objective Function:
$\\quad \\quad \\min \\quad \\sum_{i=1}^3 \\sum_{j=1}^3 c_{ij} \\cdot x_{ij}$

Constraints
1. Row Assignment Constraint:
$\\quad \\quad \\sum_{j=1}^3 x_{ij} = 1, \\quad \\forall i \\in \\{1,2,3\\}$
2. Column Assignment Constraint:
$\\quad \\quad \\sum_{i=1}^3 x_{ij} = 1, \\quad \\forall j \\in \\{1,2,3\\}$
3. Binary Constraint:
$\\quad \\quad x_{ij} \\in \\{0,1\\}, \\quad \\forall i,j$

Retrieved Information
$\\small c = \\begin{bmatrix}
3000 & 3200 & 3100 \\\\
2800 & 3300 & 2900 \\\\
2900 & 3100 & 3000 
\\end{bmatrix}$

The corresponding Python code for this instance is as follows:

```python
import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Data
c = np.array([
    [3000, 3200, 3100],
    [2800, 3300, 2900],
    [2900, 3100, 3000]
])

# Create the model
m = gp.Model("Optimization_Model")

# Decision variables
x = m.addVars(c.shape[0], c.shape[1], vtype=GRB.BINARY, name="x")

# Objective function
m.setObjective(gp.quicksum(c[i, j]*x[i, j] for i in range(c.shape[0]) for j in range(c.shape[1])), GRB.MINIMIZE)

# Constraints
for i in range(c.shape[0]):
    m.addConstr(gp.quicksum(x[i, j] for j in range(c.shape[1])) == 1, name=f"row_constraint_{i}")

for j in range(c.shape[1]):
    m.addConstr(gp.quicksum(x[i, j] for i in range(c.shape[0])) == 1, name=f"col_constraint_{j}")

# Solve the model
m.optimize()
""",
    'TP': """
For TP legacy, use the identifiers, coefficients, and dimensions in the numerical formulation.
Preserve matrix axes and orientation. Never assume a square matrix or invent entity counts.
""",
    'RA': """
For RA legacy, the complete numerical formulation and original query provide the data contract.
Use the actual identifiers and coefficients from the formulation; no canonical CSVQA_DATA exists.
Represent item counts and discrete units as nonnegative integers unless the original query explicitly
allows fractional quantities or describes divisible material. Do not infer continuity from scale alone.
Keep independent resource-by-activity allocations distinct from global activities consuming multiple
resource dimensions. A separate resource allocation requires that resource in the variable index;
a global activity consuming multiple dimensions requires the correct dimension-specific coefficients.
Every capacity constraint must use the corresponding resource decisions or consumption coefficients.
Never apply the same global expression against every independent warehouse's capacity. Preserve every
supplied capacity and per-unit coefficient, explicit business identifier, and query-supported constraint.
""",
    
}


PLANNED_CODE_INSTRUCTIONS = """
Use the formulation for model structure and Data Mapping; use CSVQA_DATA for all data.
The complete payload below is provided at execution as CSVQA_DATA. Do not define or overwrite
it, copy its rows into literals, or read local files. Derive dimensions and identifiers from its records.
Select tables by the exact table_id in Data Mapping; roles may repeat. Read records from
CSVQA_DATA["tables"][i]["records"], and read each field through record["values"][column_name].
Preserve record order and mapped axes. Do not read CSVQA_DATA["plan"].
Without a continuous pre-horizon value, start change/ramp constraints at the second period;
an initial binary state does not supply a continuous period-zero value.
Parse structured CSV text with a regex and fail explicitly if any clause is unmatched.
"""

LEGACY_CODE_INSTRUCTIONS = """
Build a self-contained program using the concrete coefficients and identifiers in the
Mathematical Optimization Model / Retrieved Information. Define all required data in the code.
Do not use CSVQA_DATA, external variables, or external files. Return executable code only.
"""

CSV_SOLVER_INSTRUCTIONS = """
For |linear expression| <= bound, use expression <= bound and expression >= -bound;
do not pass a non-variable expression to addGenConstrAbs. Use gp.quicksum, never GRB.quicksum.
Import gurobipy only as `import gurobipy as gp` and `from gurobipy import GRB`; never import `gp`.
Use only Python standard-library modules, pandas, numpy, and gurobipy; do not import a csvqa module.
Select payload tables by their exact table_id or aliases; a file_name is not itself a table_id.
Use exact supplied column names. After set_index(column), do not drop that column again.
Keep constraint names short; never put full data records in names. Never hardcode an aggregate that
can be computed from supplied rows, and bind every query-required coefficient column into the model.
Decision-variable integrality does not imply integer input coefficients: preserve decimal CSV values.
Supply/capacity is normally an upper bound while demand is a requirement. Facility flow must be linked
to its binary open variable. Use one shared flow across consecutive processing stages. For TSP, use
directed arcs with one incoming and one outgoing arc per node and a consistent subtour formulation.
Leave the solved Gurobi model in m or model. If using a function, return the model and assign it to m.
Set MIPGap=1e-6 before optimize(). Check Status before reading ObjVal. If Status == GRB.OPTIMAL,
print ObjVal and each variable's VarName and X. Otherwise print the solver status; do not report an
incumbent as optimal.
"""


def get_code(output, selected_problem):
    """Generate code for a query-only problem using the original notebook prompts."""
    route = QUERY_ONLY_CODE_ALIASES.get(selected_problem, selected_problem)
    example = QUERY_ONLY_CODE_EXAMPLES.get(route, QUERY_ONLY_CODE_EXAMPLES["Others"])
    prompt = ORIGINAL_CODE_PROMPT.format(output=output) + example
    response = make_llm().invoke([HumanMessage(content=prompt)])
    print(response.content)
    return response.content


def get_csv_code(output, route, original_query, data_payload=""):
    """Generate code from CSV-derived data: structured NRM data or a numerical formulation."""
    prompt = ORIGINAL_CODE_PROMPT.format(output=output) + f"\nOriginal Query:\n{original_query}\n"
    if data_payload:
        prompt += PLANNED_CODE_INSTRUCTIONS + f"\nComplete CSVQA_DATA:\n{data_payload}\n"
    else:
        prompt += LEGACY_CODE_INSTRUCTIONS
    prompt += CSV_CODE_GUIDANCE.get(normalize_route(route), "") + CSV_SOLVER_INSTRUCTIONS
    response = make_llm().invoke([HumanMessage(content=prompt)])
    print(response.content)
    return response.content
