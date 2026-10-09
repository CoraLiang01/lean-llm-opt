Let $x_{ij}$ denote the quantity of product shipped from warehouse $i$ to customer (store) $j$. All $x_{ij} \geq 0$ and are continuous (since the problem does not require integrality).

**Sets and Indices:**
- Warehouses $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}\}$
- Customers $j \in \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}\}$

**Parameters:**
- $d_j$: demand of customer $j$
    - $d_{\text{C1}} = 45$
    - $d_{\text{C2}} = 23$
    - $d_{\text{C3}} = 94$
    - $d_{\text{C4}} = 92$
    - $d_{\text{C5}} = 57$
    - $d_{\text{C6}} = 52$
    - $d_{\text{C7}} = 23$
    - $d_{\text{C8}} = 99$
    - $d_{\text{C9}} = 99$
    - $d_{\text{C10}} = 77$
- $s_i$: supply capacity of warehouse $i$
    - $s_{\text{S1}} = 127$
    - $s_{\text{S2}} = 236$
    - $s_{\text{S3}} = 168$
    - $s_{\text{S4}} = 115$
    - $s_{\text{S5}} = 280$
    - $s_{\text{S6}} = 179$
    - $s_{\text{S7}} = 135$
    - $s_{\text{S8}} = 263$
    - $s_{\text{S9}} = 283$
    - $s_{\text{S10}} = 476$
- $c_{ij}$: cost per unit shipped from warehouse $i$ to customer $j$ (see table below)

|        |  C1         |  C2         |  C3         |  C4         |  C5         |  C6         |  C7         |  C8         |  C9         |  C10        |
|--------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| S1     | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 169.33027   | 0.0         |
| S2     | 2077.05867  | 0.0         | 1141.04056  | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 8.06335     | 0.0         |
| S3     | 79.92103    | 474.24509   | 1477.06763  | 22.58310    | 474.24509   | 41.10660    | 474.24509   | 474.24509   | 624.16254   | 474.24509   |
| S4     | 1659.33693  | 57.20541    | 186.15190   | 1201.31371  | 1029.69746  | 41.82211    | 57.20541    | 1201.31371  | 884.56339   | 1029.69746  |
| S5     | 1297.25670  | 77.76629    | 24.26760    | 1399.79324  | 77.76629    | 53.91162    | 1399.79324  | 77.76629    | 1255.11515  | 1399.79324  |
| S6     | 1998.90907  | 985.31654   | 2.85417     | 1149.53597  | 985.31654   | 730.69236   | 54.73981    | 985.31654   | 46.80310    | 1149.53597  |
| S7     | 1780.33601  | 0.0         | 1141.04056  | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 8.06335     | 0.0         |
| S8     | 75.40936    | 1338.19873  | 21.39135    | 74.34437    | 74.34437    | 937.35062   | 1338.19873  | 1338.19873  | 1392.11866  | 1338.19873  |
| S9     | 98.90756    | 0.0         | 978.03477   | 0.0         | 0.0         | 651.11123   | 0.0         | 0.0         | 169.33027   | 0.0         |
| S10    | 2077.05867  | 0.0         | 54.33526    | 0.0         | 0.0         | 36.17285    | 0.0         | 0.0         | 145.14023   | 0.0         |

**Decision Variables:**
- $x_{ij} \geq 0$ : quantity shipped from warehouse $i$ to customer $j$ (continuous, nonnegative)

---

### Mathematical Model

**Objective:**
\[
\min \sum_{i \in \{\text{S1},\ldots,\text{S10}\}} \sum_{j \in \{\text{C1},\ldots,\text{C10}\}} c_{ij} \cdot x_{ij}
\]

**Subject to:**

**1. Demand satisfaction for each customer:**
\[
\sum_{i \in \{\text{S1},\ldots,\text{S10}\}} x_{ij} = d_j \qquad \forall j \in \{\text{C1},\ldots,\text{C10}\}
\]
That is,
\[
\begin{align*}
&\sum_{i} x_{i,\text{C1}} = 45 \\
&\sum_{i} x_{i,\text{C2}} = 23 \\
&\sum_{i} x_{i,\text{C3}} = 94 \\
&\sum_{i} x_{i,\text{C4}} = 92 \\
&\sum_{i} x_{i,\text{C5}} = 57 \\
&\sum_{i} x_{i,\text{C6}} = 52 \\
&\sum_{i} x_{i,\text{C7}} = 23 \\
&\sum_{i} x_{i,\text{C8}} = 99 \\
&\sum_{i} x_{i,\text{C9}} = 99 \\
&\sum_{i} x_{i,\text{C10}} = 77 \\
\end{align*}
\]

**2. Supply capacity for each warehouse:**
\[
\sum_{j \in \{\text{C1},\ldots,\text{C10}\}} x_{ij} \leq s_i \qquad \forall i \in \{\text{S1},\ldots,\text{S10}\}
\]
That is,
\[
\begin{align*}
&\sum_{j} x_{\text{S1},j} \leq 127 \\
&\sum_{j} x_{\text{S2},j} \leq 236 \\
&\sum_{j} x_{\text{S3},j} \leq 168 \\
&\sum_{j} x_{\text{S4},j} \leq 115 \\
&\sum_{j} x_{\text{S5},j} \leq 280 \\
&\sum_{j} x_{\text{S6},j} \leq 179 \\
&\sum_{j} x_{\text{S7},j} \leq 135 \\
&\sum_{j} x_{\text{S8},j} \leq 263 \\
&\sum_{j} x_{\text{S9},j} \leq 283 \\
&\sum_{j} x_{\text{S10},j} \leq 476 \\
\end{align*}
\]

**3. Nonnegativity:**
\[
x_{ij} \geq 0 \qquad \forall i, j
\]

**4. (Optional) Shipping only allowed where cost is defined (i.e., $c_{ij} = 0$ means shipping is not allowed):**
\[
x_{ij} = 0 \quad \text{if } c_{ij} = 0
\]

---

**All identifiers, coefficients, and constraints are as retrieved and in source order.**