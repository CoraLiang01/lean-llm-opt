Let:

- $I$ = set of warehouses, indexed by $i$ (from "Warehouse (i)" in PotentialWarehouses_Costs.csv: $I = \{1,2,3,4,5,6,7,8,9,10,11\}$)
- $J$ = set of stores, indexed by $j$ (from "Store (j)" in Stores_Demands.csv: $J = \{1,2,3,4,5,6,7,8,9,10,11\}$)
- $f_i$ = opening cost of warehouse $i$ (from "Opening Cost (fi)")
- $K_i$ = capacity of warehouse $i$ (from "Capacity (units)")
- $d_j$ = demand of store $j$ (from "Demand (units, dj)")
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to store $j$ (from TransportationCost.csv, see mapping below)
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: number of units shipped from warehouse $i$ to store $j$

#### Objective:

Minimize total cost (opening + transportation):

$$
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

#### Subject to:

1. **Demand satisfaction (each store's demand must be met):**
   $$
   \sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
   $$

2. **Warehouse capacity (cannot ship more than capacity from each warehouse):**
   $$
   \sum_{j \in J} x_{ij} \leq K_i y_i \quad \forall i \in I
   $$

3. **Variable domains:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I, j \in J
   $$
   $$
   y_i \in \{0,1\} \quad \forall i \in I
   $$

---

#### Data

**Warehouses (from PotentialWarehouses_Costs.csv, in source order):**

| Warehouse (i) | Opening Cost (fi) | Capacity (units) |
|---------------|-------------------|------------------|
| 1             | 3000              | 180              |
| 2             | 3200              | 160              |
| 3             | 3100              | 200              |
| 4             | 2800              | 150              |
| 5             | 3500              | 170              |
| 6             | 2700              | 190              |
| 7             | 2900              | 160              |
| 8             | 3050              | 175              |
| 9             | 3100              | 170              |
| 10            | 2200              | 180              |
| 11            | 2890              | 190              |

**Stores (from Stores_Demands.csv, in source order):**

| Store (j) | Demand (units, dj) |
|-----------|--------------------|
| 1         | 30                 |
| 2         | 40                 |
| 3         | 20                 |
| 4         | 35                 |
| 5         | 20                 |
| 6         | 25                 |
| 7         | 45                 |
| 8         | 38                 |
| 9         | 32                 |
| 10        | 41                 |
| 11        | 44                 |

**Transportation Costs $c_{ij}$ (from TransportationCost.csv, in source order):**

Let $c_{ij}$ be the cost per unit from warehouse $i$ to store $j$.

- The rows correspond to warehouses $i=1$ to $11$ (W1 to W11).
- The columns correspond to stores $j=1$ to $11$ (W1 to W11).

| $c_{ij}$ | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|----------|----|----|----|----|----|----|----|----|----|----|----|
| 1        | 12 | 11 | 14 | 15 | 17 | 13 | 12 | 16 | 16 | 14 | 15 |
| 2        | 17 | 19 | 15 | 20 | 18 | 14 | 17 | 15 | 13 | 15 | 16 |
| 3        | 13 | 14 | 12 | 14 | 16 | 15 | 11 | 14 | 16 | 18 | 17 |
| 4        | 18 | 16 | 17 | 13 | 18 | 17 | 14 | 19 | 16 | 13 | 18 |
| 5        | 10 | 13 | 12 | 19 | 15 | 11 | 12 | 14 | 12 | 15 | 17 |
| 6        | 15 | 12 | 14 | 16 | 13 | 17 | 16 | 16 | 14 | 18 | 19 |
| 7        | 14 | 13 | 15 | 17 | 12 | 13 | 14 | 15 | 12 | 16 | 14 |
| 8        | 19 | 16 | 18 | 20 | 17 | 19 | 16 | 18 | 15 | 15 | 18 |
| 9        | 17 | 18 | 12 | 14 | 16 | 15 | 14 | 17 | 21 | 15 | 18 |
| 10       | 14 | 13 | 15 | 17 | 16 | 18 | 14 | 19 | 15 | 17 | 19 |
| 11       | 15 | 13 | 16 | 17 | 11 | 13 | 14 | 15 | 19 | 21 | 13 |

- For example, $c_{3,5} = 16$ (warehouse 3 to store 5), $c_{7,2} = 13$ (warehouse 7 to store 2).

---

**Complete Model:**

$$
\begin{align*}
\min \quad & \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij} \\
\text{s.t.} \quad & \sum_{i=1}^{11} x_{ij} = d_j \quad \forall j=1,\ldots,11 \\
& \sum_{j=1}^{11} x_{ij} \leq K_i y_i \quad \forall i=1,\ldots,11 \\
& x_{ij} \geq 0 \quad \forall i=1,\ldots,11;\ j=1,\ldots,11 \\
& y_i \in \{0,1\} \quad \forall i=1,\ldots,11 \\
\end{align*}
$$

with all coefficients and indices as given above.