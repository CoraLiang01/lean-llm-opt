Let $I = \{A1, A2, \ldots, A80\}$ index the 80 products.

Define for each product $i \in I$:
- $x_i$: integer number of 100 kg units of product $i$ to produce in the month ($x_i \in \mathbb{Z}_{\geq 0}$)
- $y_i$: binary variable, $y_i = 1$ if product $i$'s production line is activated, $0$ otherwise ($y_i \in \{0,1\}$)

Parameters (all indexed by $i$ as below):

- $d_i$: Maximum Demand (100 kg units)
- $p_i$: Selling Price ($/100$ kg)
- $c_i$: Production Cost ($/100$ kg)
- $q_i$: Production Quota (max per day, 100 kg units)
- $f_i$: Activation Cost ($)
- $m_i$: Minimum Batch Size (100 kg units)
- $T = 22$: Number of production days in the month

Data (all values as given in the CSVs):

| Product | $d_i$ | $p_i$ | $c_i$ | $q_i$ | $f_i$ | $m_i$ |
|---------|-------|-------|-------|-------|-------|-------|
| A1      | 5732  | 111   | 71.38 | 586   | 178539| 18    |
| A2      | 5607  | 81    | 45.02 | 329   | 157708| 25    |
| ...     | ...   | ...   | ...   | ...   | ...   | ...   |
| A80     | 4466  | 106   | 67.19 | 556   | 146842| 18    |

(Full data as in the retrieved CSVs, in original order.)

The complete model is:

**Objective:**
\[
\max \sum_{i \in I} \left[ (p_i - c_i) x_i - f_i y_i \right]
\]

**Subject to:**

1. **Demand constraint (do not exceed demand):**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Production quota constraint (do not exceed total monthly capacity):**
   \[
   x_i \leq q_i \cdot T \qquad \forall i \in I
   \]

3. **Minimum batch size constraint (if produced, must meet minimum batch):**
   \[
   x_i \geq m_i y_i \qquad \forall i \in I
   \]

4. **Linking constraint (cannot produce unless activated):**
   \[
   x_i \leq (q_i \cdot T) y_i \qquad \forall i \in I
   \]
   (This is redundant with constraint 2, but ensures $x_i = 0$ if $y_i = 0$.)

5. **Variable domains:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]
   \[
   y_i \in \{0,1\} \qquad \forall i \in I
   \]

**Parameters (from CSVs, in original order):**

- For $i = $A1 to A80:
    - $d_i$ = Maximum Demand (100 kg units):  
      5732, 5607, 4653, ..., 4466
    - $p_i$ = Selling Price ($/100$ kg):  
      111, 81, 145, ..., 106
    - $c_i$ = Production Cost ($/100$ kg):  
      71.38, 45.02, 107.04, ..., 67.19
    - $q_i$ = Production Quota (max per day):  
      586, 329, 541, ..., 556
    - $f_i$ = Activation Cost ($):  
      178539, 157708, 85192, ..., 146842
    - $m_i$ = Minimum Batch Size (100 kg units):  
      18, 25, 23, ..., 18

**Summary Table (first and last product shown for illustration):**

| Product | $d_i$ | $p_i$ | $c_i$ | $q_i$ | $f_i$ | $m_i$ |
|---------|-------|-------|-------|-------|-------|-------|
| A1      | 5732  | 111   | 71.38 | 586   | 178539| 18    |
| ...     | ...   | ...   | ...   | ...   | ...   | ...   |
| A80     | 4466  | 106   | 67.19 | 556   | 146842| 18    |

**Decision variables:**
- $x_i$: integer number of 100 kg units of product $i$ to produce in the month
- $y_i$: binary, 1 if product $i$'s line is activated, 0 otherwise

**All constraints and coefficients are as above, using the original CSV order and values.**