Let:

- $I$ = set of warehouses, indexed by $i$ (from "Warehouse (i)" in PotentialWarehouses_Costs.csv: $I = \{1,2,3,4,5,6,7,8,9,10,11\}$)
- $J$ = set of stores, indexed by $j$ (from "Store (j)" in Stores_Demands.csv: $J = \{1,2,3,4,5,6,7,8,9,10,11\}$)
- $f_i$ = opening cost of warehouse $i$
- $K_i$ = capacity of warehouse $i$
- $d_j$ = demand of store $j$
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to store $j$
- $y_i \in \{0,1\}$: 1 if warehouse $i$ is opened, 0 otherwise
- $x_{ij} \geq 0$: units supplied from warehouse $i$ to store $j$

#### Parameters (from data):

- $f_1=3000$, $K_1=180$; $f_2=3200$, $K_2=160$; $f_3=3100$, $K_3=200$; $f_4=2800$, $K_4=150$; $f_5=3500$, $K_5=170$; $f_6=2700$, $K_6=190$; $f_7=2900$, $K_7=160$; $f_8=3050$, $K_8=175$; $f_9=3100$, $K_9=170$; $f_{10}=2200$, $K_{10}=180$; $f_{11}=2890$, $K_{11}=190$
- $d_1=30$, $d_2=40$, $d_3=20$, $d_4=35$, $d_5=20$, $d_6=25$, $d_7=45$, $d_8=38$, $d_9=32$, $d_{10}=41$, $d_{11}=44$
- $c_{ij}$ as below (rows: $i=1$ to $11$; columns: $j=1$ to $11$):

| $c_{ij}$ | $j=1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ | $8$ | $9$ | $10$ | $11$ |
|----------|-------|-----|-----|-----|-----|-----|-----|-----|-----|------|------|
| $i=1$    | 12    | 11  | 14  | 15  | 17  | 13  | 12  | 16  | 16  | 14   | 15   |
| $2$      | 17    | 19  | 15  | 20  | 18  | 14  | 17  | 15  | 13  | 15   | 16   |
| $3$      | 13    | 14  | 12  | 14  | 16  | 15  | 11  | 14  | 16  | 18   | 17   |
| $4$      | 18    | 16  | 17  | 13  | 18  | 17  | 14  | 19  | 16  | 13   | 18   |
| $5$      | 10    | 13  | 12  | 19  | 15  | 11  | 12  | 14  | 12  | 15   | 17   |
| $6$      | 15    | 12  | 14  | 16  | 13  | 17  | 16  | 16  | 14  | 18   | 19   |
| $7$      | 14    | 13  | 15  | 17  | 12  | 13  | 14  | 15  | 12  | 16   | 14   |
| $8$      | 19    | 16  | 18  | 20  | 17  | 19  | 16  | 18  | 15  | 15   | 18   |
| $9$      | 17    | 18  | 12  | 14  | 16  | 15  | 14  | 17  | 21  | 15   | 18   |
| $10$     | 14    | 13  | 15  | 17  | 16  | 18  | 14  | 19  | 15  | 17   | 19   |
| $11$     | 15    | 13  | 16  | 17  | 11  | 13  | 14  | 15  | 19  | 21   | 13   |

#### Mathematical Model

**Decision Variables:**
- $y_i \in \{0,1\}$, $\forall i \in I$
- $x_{ij} \geq 0$, $\forall i \in I, j \in J$

**Objective:**
\[
\min \sum_{i=1}^{11} f_i y_i + \sum_{i=1}^{11} \sum_{j=1}^{11} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i=1}^{11} x_{ij} = d_j, \quad \forall j = 1,\ldots,11
   \]

2. **Warehouse capacity (cannot exceed capacity if open, zero if closed):**
   \[
   \sum_{j=1}^{11} x_{ij} \leq K_i y_i, \quad \forall i = 1,\ldots,11
   \]

3. **Variable domains:**
   \[
   y_i \in \{0,1\}, \quad \forall i = 1,\ldots,11
   \]
   \[
   x_{ij} \geq 0, \quad \forall i = 1,\ldots,11, \; j = 1,\ldots,11
   \]

**Parameter values (from data):**

- $f_1=3000$, $K_1=180$; $f_2=3200$, $K_2=160$; $f_3=3100$, $K_3=200$; $f_4=2800$, $K_4=150$; $f_5=3500$, $K_5=170$; $f_6=2700$, $K_6=190$; $f_7=2900$, $K_7=160$; $f_8=3050$, $K_8=175$; $f_9=3100$, $K_9=170$; $f_{10}=2200$, $K_{10}=180$; $f_{11}=2890$, $K_{11}=190$
- $d_1=30$, $d_2=40$, $d_3=20$, $d_4=35$, $d_5=20$, $d_6=25$, $d_7=45$, $d_8=38$, $d_9=32$, $d_{10}=41$, $d_{11}=44$
- $c_{ij}$ as in the table above.

**All indices, coefficients, and constraints are as retrieved and required by the query.**