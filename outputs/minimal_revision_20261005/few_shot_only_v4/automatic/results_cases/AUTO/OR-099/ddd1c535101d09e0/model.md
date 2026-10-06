##### Decision Variables

- $x_{ij} \geq 0$: Amount shipped from warehouse $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise (binary).

##### Parameters

- $I = \{1,2,3,4,5,6,7,8,9,10,11\}$: Set of warehouses.
- $J = \{1,2,3,4,5,6,7,8,9,10,11\}$: Set of stores.
- $f_i$: Opening cost for warehouse $i$.
- $K_i$: Capacity of warehouse $i$.
- $d_j$: Demand of store $j$.
- $c_{ij}$: Transportation cost per unit from warehouse $i$ to store $j$.

##### Objective Function

\[
\min \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} = d_j \qquad \forall j \in J
   \]

2. **Warehouse capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq K_i y_i \qquad \forall i \in I
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Warehouses ($I$):** $1,2,3,4,5,6,7,8,9,10,11$  
  (from column "Warehouse (i)" in PotentialWarehouses_Costs.csv)

- **Stores ($J$):** $1,2,3,4,5,6,7,8,9,10,11$  
  (from column "Store (j)" in Stores_Demands.csv)

- **Warehouse opening costs ($f_i$) and capacities ($K_i$):**  
  (from PotentialWarehouses_Costs.csv, columns "Opening Cost (fi)" and "Capacity (units)")
  - $f_1 = 3000$, $K_1 = 180$
  - $f_2 = 3200$, $K_2 = 160$
  - $f_3 = 3100$, $K_3 = 200$
  - $f_4 = 2800$, $K_4 = 150$
  - $f_5 = 3500$, $K_5 = 170$
  - $f_6 = 2700$, $K_6 = 190$
  - $f_7 = 2900$, $K_7 = 160$
  - $f_8 = 3050$, $K_8 = 175$
  - $f_9 = 3100$, $K_9 = 170$
  - $f_{10} = 2200$, $K_{10} = 180$
  - $f_{11} = 2890$, $K_{11} = 190$

- **Store demands ($d_j$):**  
  (from Stores_Demands.csv, column "Demand (units, dj)")
  - $d_1 = 30$
  - $d_2 = 40$
  - $d_3 = 20$
  - $d_4 = 35$
  - $d_5 = 20$
  - $d_6 = 25$
  - $d_7 = 45$
  - $d_8 = 38$
  - $d_9 = 32$
  - $d_{10} = 41$
  - $d_{11} = 44$

- **Transportation costs ($c_{ij}$):**  
  (from TransportationCost.csv, $c_{ij}$ is the cost from warehouse $i$ to store $j$; warehouse and store indices correspond to $W1$–$W11$ and $1$–$11$ respectively. The matrix is as follows, with $i$ as row and $j$ as column):

| $c_{ij}$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|----------|---|---|---|---|---|---|---|---|---|----|----|
| **1**    |12 |11 |14 |15 |17 |13 |12 |16 |16 |14  |15  |
| **2**    |17 |19 |15 |20 |18 |14 |17 |15 |13 |15  |16  |
| **3**    |13 |14 |12 |14 |16 |15 |11 |14 |16 |18  |17  |
| **4**    |18 |16 |17 |13 |18 |17 |14 |19 |16 |13  |18  |
| **5**    |10 |13 |12 |19 |15 |11 |12 |14 |12 |15  |17  |
| **6**    |15 |12 |14 |16 |13 |17 |16 |16 |14 |18  |19  |
| **7**    |14 |13 |15 |17 |12 |13 |14 |15 |12 |16  |14  |
| **8**    |19 |16 |18 |20 |17 |19 |16 |18 |15 |15  |18  |
| **9**    |17 |18 |12 |14 |16 |15 |14 |17 |21 |15  |18  |
| **10**   |14 |13 |15 |17 |16 |18 |14 |19 |15 |17  |19  |
| **11**   |15 |13 |16 |17 |11 |13 |14 |15 |19 |21  |13  |

- **Source-column mapping:**  
  - Warehouse indices $i$ correspond to "Warehouse (i)" in PotentialWarehouses_Costs.csv and "W1"–"W11" in TransportationCost.csv.
  - Store indices $j$ correspond to "Store (j)" in Stores_Demands.csv and columns $1$–$11$ in TransportationCost.csv.

---

**This model determines which warehouses to open and how to allocate shipments to stores to minimize total cost, subject to demand, capacity, and logical constraints.**