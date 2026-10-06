##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i \in I$ to customer $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I = \{$W1, W2, W3, W4, W5, W6, W7, W8, W9, W10$\}$ (warehouses)
- $J = \{$C1, C2, ..., C20$\}$ (customers)
- $f_i$: Fixed cost of opening warehouse $i$ (from warehouse.csv)
- $u_i$: Capacity of warehouse $i$ (from warehouse.csv)
- $d_j$: Demand of customer $j$ (from demand.csv)
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to customer $j$ (from cost.csv)

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Warehouse capacity:**  
   For each warehouse $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq u_i y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Warehouses ($I$):**  
  W1, W2, W3, W4, W5, W6, W7, W8, W9, W10  
  (from warehouse.csv, cost.csv)

- **Customers ($J$):**  
  C1, C2, ..., C20  
  (from demand.csv, cost.csv)

- **Fixed costs ($f_i$) and capacities ($u_i$):**  
  From warehouse.csv:  
  | Warehouse | Fixed_Cost | Capacity |
  |-----------|------------|----------|
  | W1        | 2000       | 1000     |
  | W2        | 2500       | 1500     |
  | W3        | 1800       | 1200     |
  | W4        | 3200       | 2000     |
  | W5        | 1500       | 800      |
  | W6        | 4000       | 2500     |
  | W7        | 2800       | 1800     |
  | W8        | 1950       | 1100     |
  | W9        | 3500       | 2100     |
  | W10       | 2200       | 1300     |

- **Customer demands ($d_j$):**  
  From demand.csv:  
  | Customer | Demand |
  |----------|--------|
  | C1       | 800    |
  | C2       | 600    |
  | C3       | 500    |
  | C4       | 700    |
  | C5       | 450    |
  | C6       | 950    |
  | C7       | 350    |
  | C8       | 850    |
  | C9       | 400    |
  | C10      | 750    |
  | C11      | 900    |
  | C12      | 550    |
  | C13      | 650    |
  | C14      | 820    |
  | C15      | 480    |
  | C16      | 920    |
  | C17      | 320    |
  | C18      | 780    |
  | C19      | 520    |
  | C20      | 680    |

- **Transportation costs ($c_{ij}$):**  
  From cost.csv:  
  Each row corresponds to a warehouse (W1–W10), each column to a customer (C1–C20).  
  For example, $c_{W1,C1} = 10$, $c_{W2,C3} = 9$, etc.

---

**All parameters and indices are directly mapped from the provided CSV columns and rows.**