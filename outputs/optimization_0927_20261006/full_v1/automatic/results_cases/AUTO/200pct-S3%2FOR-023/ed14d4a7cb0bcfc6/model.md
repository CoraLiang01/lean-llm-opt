##### Decision Variables

- $x_{ij} \geq 0$: Quantity of goods shipped from supplier $i \in I$ to store $j \in J$ (continuous).
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise.

##### Sets

- Suppliers $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- Stores $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$

##### Parameters

- Store demands $d_j$:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Supplier fixed costs $f_i$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Transportation costs $c_{ij}$ (supplier $i$ to store $j$):

| Supplier      | Customer_1 (CLARINDA) | Customer_2 (FORT MADISON) | Customer_3 (SIOUX CITY) | Customer_4 (TOLEDO) | Customer_5 (BANCROFT) |
|---------------|-----------------------|---------------------------|-------------------------|---------------------|-----------------------|
| MOUNT AYR     | 694.68                | 17.48                     | 20.07                   | 199.02              | 1685.53               |
| WAUKEE        | 15.13                 | 1.5                       | 1.43                    | 27.88               | 90.69                 |
| WAVERLY       | 2.34                  | 349.34                    | 246.6                   | 41.3                | 78.73                 |
| PELLA         | 1181.6                | 1458.53                   | 1646.36                 | 1924.55             | 38.93                 |
| DES MOINES    | 1030.8                | 43.48                     | 932.43                  | 55.39               | 103.84                |

##### Mathematical Model

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]

2. **Supplier activation (no shipments from inactive suppliers):**
   \[
   \sum_{j \in J} x_{ij} \leq M y_i, \quad \forall i \in I
   \]
   where $M = \sum_{j \in J} d_j = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

##### Parameters (full vectors and matrices):

- $I = \{\text{MOUNT AYR}, \text{WAUKEE}, \text{WAVERLY}, \text{PELLA}, \text{DES MOINES}\}$
- $J = \{\text{Customer\_1}, \text{Customer\_2}, \text{Customer\_3}, \text{Customer\_4}, \text{Customer\_5}\}$
- $d = [2397, 1889, 2518, 3218, 1813]$
- $f = [96.58, 94.06, 94.37, 82.88, 94.96]$
- $C =$
\[
\begin{bmatrix}
694.68 & 17.48 & 20.07 & 199.02 & 1685.53 \\
15.13 & 1.5 & 1.43 & 27.88 & 90.69 \\
2.34 & 349.34 & 246.6 & 41.3 & 78.73 \\
1181.6 & 1458.53 & 1646.36 & 1924.55 & 38.93 \\
1030.8 & 43.48 & 932.43 & 55.39 & 103.84 \\
\end{bmatrix}
\]
with rows in the order: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES and columns in the order: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5.

- $M = 11835$

---

This model determines which suppliers to activate and how much each should ship to each store to minimize total cost while meeting all store demands.