Let:

- $I$ = set of warehouses: $\{W1, W2, W3, W4, W5, W6, W7, W8, W9, W10\}$
- $J$ = set of customers: $\{C1, C2, ..., C20\}$
- $f_i$ = fixed cost of opening warehouse $i$
- $s_i$ = capacity of warehouse $i$
- $d_j$ = demand of customer $j$
- $c_{ij}$ = unit transportation cost from warehouse $i$ to customer $j$
- $x_{ij}$ = amount supplied from warehouse $i$ to customer $j$ (continuous, $\geq 0$)
- $y_i$ = 1 if warehouse $i$ is opened, 0 otherwise (binary)

Parameters (from data):

Warehouses (from warehouse.csv):

| Warehouse ID | Fixed_Cost | Capacity |
|--------------|------------|----------|
| W1           | 2000       | 1000     |
| W2           | 2500       | 1500     |
| W3           | 1800       | 1200     |
| W4           | 3200       | 2000     |
| W5           | 1500       | 800      |
| W6           | 4000       | 2500     |
| W7           | 2800       | 1800     |
| W8           | 1950       | 1100     |
| W9           | 3500       | 2100     |
| W10          | 2200       | 1300     |

Customers (from demand.csv):

| Customer ID | Demand |
|-------------|--------|
| C1          | 800    |
| C2          | 600    |
| C3          | 500    |
| C4          | 700    |
| C5          | 450    |
| C6          | 950    |
| C7          | 350    |
| C8          | 850    |
| C9          | 400    |
| C10         | 750    |
| C11         | 900    |
| C12         | 550    |
| C13         | 650    |
| C14         | 820    |
| C15         | 480    |
| C16         | 920    |
| C17         | 320    |
| C18         | 780    |
| C19         | 520    |
| C20         | 680    |

Transportation costs $c_{ij}$ (from cost.csv):

- For each warehouse $i$ and customer $j$, $c_{ij}$ is given by the corresponding entry in the cost.csv table.

##### Mathematical Model

**Decision Variables:**
- $x_{ij} \geq 0$ (continuous): amount shipped from warehouse $i$ to customer $j$
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise

**Objective:**
\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (every customer is fully served):**
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]

2. **Warehouse capacity (cannot exceed capacity if open, zero if closed):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i y_i \qquad \forall i \in I
   \]

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

**Parameter values:**

- $f_i$ and $s_i$ as given in warehouse.csv above.
- $d_j$ as given in demand.csv above.
- $c_{ij}$ as given in cost.csv, e.g., $c_{W1,C1} = 10$, $c_{W2,C1} = 18$, etc.

**All indices, costs, demands, and capacities are as retrieved and must be used as shown.**