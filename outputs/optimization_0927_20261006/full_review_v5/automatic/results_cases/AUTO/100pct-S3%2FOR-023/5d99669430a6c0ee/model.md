##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i$ to store $j$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary).

##### Parameters

**Suppliers ($I$):**
- MOUNT AYR
- WAUKEE
- WAVERLY
- PELLA
- DES MOINES

**Stores / Customers ($J$):**
- Customer_1
- Customer_2
- Customer_3
- Customer_4
- Customer_5

**Store Demands ($d_j$):**
| Store        | Demand |
|--------------|--------|
| Customer_1   | 2397   |
| Customer_2   | 1889   |
| Customer_3   | 2518   |
| Customer_4   | 3218   |
| Customer_5   | 1813   |

**Supplier Fixed Costs ($f_i$):**
| Supplier      | Fixed Cost |
|--------------|------------|
| MOUNT AYR    | 96.58      |
| WAUKEE       | 94.06      |
| WAVERLY      | 94.37      |
| PELLA        | 82.88      |
| DES MOINES   | 94.96      |

**Transportation Costs ($c_{ij}$):**

| Supplier    | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|-------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR   | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE      | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY     | 2.34                  | 349.34                    | 246.6                   | 41.3                | 78.73                 |
| PELLA       | 1181.6                | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES  | 1030.8                | 43.48                     | 932.43                  | 55.39               | 103.84                |

**Big-M parameter ($M$):**
- $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

##### Objective Function

\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   \[
   \sum_{i \in I} x_{ij} = d_j
   \]

2. **Supplier activation:**  
   For each supplier $i \in I$,
   \[
   \sum_{j \in J} x_{ij} \leq M y_i
   \]

3. **Variable domains:**  
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]
   \[
   y_i \in \{0,1\} \quad \forall i \in I
   \]

##### Parameter Tables

**Supplier Set ($I$):**
- MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES

**Store Set ($J$):**
- Customer_1, Customer_2, Customer_3, Customer_4, Customer_5

**Demands ($d_j$):**
| Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|------------|------------|------------|------------|------------|
| 2397       | 1889       | 2518       | 3218       | 1813       |

**Fixed Costs ($f_i$):**
| MOUNT AYR | WAUKEE | WAVERLY | PELLA | DES MOINES |
|-----------|--------|---------|-------|------------|
| 96.58     | 94.06  | 94.37   | 82.88 | 94.96      |

**Transportation Costs ($c_{ij}$):**
|             | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|-------------|------------|------------|------------|------------|------------|
| MOUNT AYR   | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE      | 15.13      | 1.5        | 1.43       | 27.88      | 90.69      |
| WAVERLY     | 2.34       | 349.34     | 246.6      | 41.3       | 78.73      |
| PELLA       | 1181.6     | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES  | 1030.8     | 43.48      | 932.43     | 55.39      | 103.84     |

**Big-M ($M$):** 11835

---

**Summary:**  
This model determines which suppliers to activate and how much each should ship to each store, so that all store demands are met at minimum total cost (transportation plus fixed supplier opening costs). Each supplier can only ship if activated, and there are no explicit supplier capacity limits. All parameters are as retrieved from the provided CSV files.