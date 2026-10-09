##### Sets

- $I = \{\text{MOUNT AYR},\ \text{WAUKEE},\ \text{WAVERLY},\ \text{PELLA},\ \text{DES MOINES}\}$ (Suppliers)
- $J = \{\text{Customer\_1},\ \text{Customer\_2},\ \text{Customer\_3},\ \text{Customer\_4},\ \text{Customer\_5}\}$ (Stores)

##### Parameters

- Fixed costs for each supplier $i \in I$:
  - $f_{\text{MOUNT AYR}} = 96.58$
  - $f_{\text{WAUKEE}} = 94.06$
  - $f_{\text{WAVERLY}} = 94.37$
  - $f_{\text{PELLA}} = 82.88$
  - $f_{\text{DES MOINES}} = 94.96$

- Demand for each store $j \in J$:
  - $d_{\text{Customer\_1}} = 2397$
  - $d_{\text{Customer\_2}} = 1889$
  - $d_{\text{Customer\_3}} = 2518$
  - $d_{\text{Customer\_4}} = 3218$
  - $d_{\text{Customer\_5}} = 1813$

- Transportation costs $c_{ij}$ from supplier $i$ to store $j$:

| Supplier $\downarrow$ \ Store $\rightarrow$ | Customer_1 | Customer_2 | Customer_3 | Customer_4 | Customer_5 |
|---------------------------------------------|------------|------------|------------|------------|------------|
| MOUNT AYR                                   | 694.68     | 17.48      | 20.07      | 199.02     | 1685.53    |
| WAUKEE                                      | 15.13      | 1.50       | 1.43       | 27.88      | 90.69      |
| WAVERLY                                     | 2.34       | 349.34     | 246.60     | 41.30      | 78.73      |
| PELLA                                       | 1181.60    | 1458.53    | 1646.36    | 1924.55    | 38.93      |
| DES MOINES                                  | 1030.80    | 43.48      | 932.43     | 55.39      | 103.84     |

##### Decision Variables

- $x_{ij} \geq 0$: Quantity shipped from supplier $i$ to store $j$ (continuous)
- $y_i \in \{0,1\}$: 1 if supplier $i$ is activated (open), 0 otherwise (binary)

##### Objective Function

\[
\min \left( \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \right)
\]

That is,

\[
\min \Bigg(
\begin{aligned}
&694.68\,x_{\text{MOUNT AYR},\text{Customer\_1}} + 17.48\,x_{\text{MOUNT AYR},\text{Customer\_2}} + 20.07\,x_{\text{MOUNT AYR},\text{Customer\_3}} + 199.02\,x_{\text{MOUNT AYR},\text{Customer\_4}} + 1685.53\,x_{\text{MOUNT AYR},\text{Customer\_5}} \\
+& 15.13\,x_{\text{WAUKEE},\text{Customer\_1}} + 1.50\,x_{\text{WAUKEE},\text{Customer\_2}} + 1.43\,x_{\text{WAUKEE},\text{Customer\_3}} + 27.88\,x_{\text{WAUKEE},\text{Customer\_4}} + 90.69\,x_{\text{WAUKEE},\text{Customer\_5}} \\
+& 2.34\,x_{\text{WAVERLY},\text{Customer\_1}} + 349.34\,x_{\text{WAVERLY},\text{Customer\_2}} + 246.60\,x_{\text{WAVERLY},\text{Customer\_3}} + 41.30\,x_{\text{WAVERLY},\text{Customer\_4}} + 78.73\,x_{\text{WAVERLY},\text{Customer\_5}} \\
+& 1181.60\,x_{\text{PELLA},\text{Customer\_1}} + 1458.53\,x_{\text{PELLA},\text{Customer\_2}} + 1646.36\,x_{\text{PELLA},\text{Customer\_3}} + 1924.55\,x_{\text{PELLA},\text{Customer\_4}} + 38.93\,x_{\text{PELLA},\text{Customer\_5}} \\
+& 1030.80\,x_{\text{DES MOINES},\text{Customer\_1}} + 43.48\,x_{\text{DES MOINES},\text{Customer\_2}} + 932.43\,x_{\text{DES MOINES},\text{Customer\_3}} + 55.39\,x_{\text{DES MOINES},\text{Customer\_4}} + 103.84\,x_{\text{DES MOINES},\text{Customer\_5}} \\
+& 96.58\,y_{\text{MOUNT AYR}} + 94.06\,y_{\text{WAUKEE}} + 94.37\,y_{\text{WAVERLY}} + 82.88\,y_{\text{PELLA}} + 94.96\,y_{\text{DES MOINES}}
\end{aligned}
\Bigg)
\]

##### Constraints

1. **Demand satisfaction:** Each store’s demand must be met exactly:
   \[
   \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J
   \]
   That is,
   - $x_{\text{MOUNT AYR},\text{Customer\_1}} + x_{\text{WAUKEE},\text{Customer\_1}} + x_{\text{WAVERLY},\text{Customer\_1}} + x_{\text{PELLA},\text{Customer\_1}} + x_{\text{DES MOINES},\text{Customer\_1}} = 2397$
   - $x_{\text{MOUNT AYR},\text{Customer\_2}} + x_{\text{WAUKEE},\text{Customer\_2}} + x_{\text{WAVERLY},\text{Customer\_2}} + x_{\text{PELLA},\text{Customer\_2}} + x_{\text{DES MOINES},\text{Customer\_2}} = 1889$
   - $x_{\text{MOUNT AYR},\text{Customer\_3}} + x_{\text{WAUKEE},\text{Customer\_3}} + x_{\text{WAVERLY},\text{Customer\_3}} + x_{\text{PELLA},\text{Customer\_3}} + x_{\text{DES MOINES},\text{Customer\_3}} = 2518$
   - $x_{\text{MOUNT AYR},\text{Customer\_4}} + x_{\text{WAUKEE},\text{Customer\_4}} + x_{\text{WAVERLY},\text{Customer\_4}} + x_{\text{PELLA},\text{Customer\_4}} + x_{\text{DES MOINES},\text{Customer\_4}} = 3218$
   - $x_{\text{MOUNT AYR},\text{Customer\_5}} + x_{\text{WAUKEE},\text{Customer\_5}} + x_{\text{WAVERLY},\text{Customer\_5}} + x_{\text{PELLA},\text{Customer\_5}} + x_{\text{DES MOINES},\text{Customer\_5}} = 1813$

2. **Supplier activation:** No shipments from a supplier unless it is open. Since there are no explicit supplier capacity limits, use a big-M constraint with $M = \sum_{j \in J} d_j = 11835$:
   \[
   \sum_{j \in J} x_{ij} \leq M\, y_i, \quad \forall i \in I
   \]
   That is, for each supplier $i$:
   - $x_{i,\text{Customer\_1}} + x_{i,\text{Customer\_2}} + x_{i,\text{Customer\_3}} + x_{i,\text{Customer\_4}} + x_{i,\text{Customer\_5}} \leq 11835\, y_i$

3. **Variable domains:**
   \[
   x_{ij} \geq 0 \quad \text{(continuous)}, \quad y_i \in \{0,1\}
   \]

---

##### Summary of Parameters

- Suppliers: MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES
- Stores: Customer_1, Customer_2, Customer_3, Customer_4, Customer_5
- Fixed costs: as listed above
- Demands: as listed above
- Transportation cost matrix: as listed above
- $M = 2397 + 1889 + 2518 + 3218 + 1813 = 11835$

---

##### Complete Mathematical Model

\[
\begin{align*}
\min\ & \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij} + \sum_{i \in I} f_i y_i \\
\text{s.t.}\quad
& \sum_{i \in I} x_{ij} = d_j, \quad \forall j \in J \\
& \sum_{j \in J} x_{ij} \leq 11835\, y_i, \quad \forall i \in I \\
& x_{ij} \geq 0,\quad \forall i \in I,\, j \in J \\
& y_i \in \{0,1\},\quad \forall i \in I
\end{align*}
\]

with all parameters and sets as specified above.