Let $x_{ij}$ denote the quantity of product shipped from warehouse (region) $i$ to store (customer) $j$. All $x_{ij} \geq 0$ and can be fractional.

**Sets:**
- Warehouses (regions): $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Stores (customers): $j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

**Parameters:**
- Demand at store $j$ ($d_j$):

  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$

- Supply capacity at warehouse $i$ ($s_i$):

  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$

- Transportation cost per unit from warehouse $i$ to store $j$ ($c_{ij}$):

  |        | D1                | D2                | D3                | D4                | D5                |
  |--------|-------------------|-------------------|-------------------|-------------------|-------------------|
  | S1     | 269.3910588020795 | 1.4537335390933939| 99.60345345756605 | 26.64078166309837 | 9.537688956880922 |
  | S2     | 9.291846876785183 | 10.874778437070223| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
  | S3     | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.2121910887916876| 133.8493396124168 |
  | S4     | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
  | S5     | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|

**Decision Variables:**
- $x_{ij} \geq 0$ (continuous), for all $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ and $j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

---

### Mathematical Model

**Objective:**
\[
\min \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}} \sum_{j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}} c_{ij} \, x_{ij}
\]

**Subject to:**

1. **Demand satisfaction at each store:**
   \[
   \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}} x_{ij} = d_j, \quad \forall j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}
   \]
   Explicitly:
   \begin{align*}
   x_{\text{S1},\text{D1}} + x_{\text{S2},\text{D1}} + x_{\text{S3},\text{D1}} + x_{\text{S4},\text{D1}} + x_{\text{S5},\text{D1}} &= 428 \\
   x_{\text{S1},\text{D2}} + x_{\text{S2},\text{D2}} + x_{\text{S3},\text{D2}} + x_{\text{S4},\text{D2}} + x_{\text{S5},\text{D2}} &= 217 \\
   x_{\text{S1},\text{D3}} + x_{\text{S2},\text{D3}} + x_{\text{S3},\text{D3}} + x_{\text{S4},\text{D3}} + x_{\text{S5},\text{D3}} &= 214 \\
   x_{\text{S1},\text{D4}} + x_{\text{S2},\text{D4}} + x_{\text{S3},\text{D4}} + x_{\text{S4},\text{D4}} + x_{\text{S5},\text{D4}} &= 380 \\
   x_{\text{S1},\text{D5}} + x_{\text{S2},\text{D5}} + x_{\text{S3},\text{D5}} + x_{\text{S4},\text{D5}} + x_{\text{S5},\text{D5}} &= 254 \\
   \end{align*}

2. **Supply capacity at each warehouse:**
   \[
   \sum_{j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}} x_{ij} \leq s_i, \quad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}
   \]
   Explicitly:
   \begin{align*}
   x_{\text{S1},\text{D1}} + x_{\text{S1},\text{D2}} + x_{\text{S1},\text{D3}} + x_{\text{S1},\text{D4}} + x_{\text{S1},\text{D5}} &\leq 428 \\
   x_{\text{S2},\text{D1}} + x_{\text{S2},\text{D2}} + x_{\text{S2},\text{D3}} + x_{\text{S2},\text{D4}} + x_{\text{S2},\text{D5}} &\leq 217 \\
   x_{\text{S3},\text{D1}} + x_{\text{S3},\text{D2}} + x_{\text{S3},\text{D3}} + x_{\text{S3},\text{D4}} + x_{\text{S3},\text{D5}} &\leq 214 \\
   x_{\text{S4},\text{D1}} + x_{\text{S4},\text{D2}} + x_{\text{S4},\text{D3}} + x_{\text{S4},\text{D4}} + x_{\text{S4},\text{D5}} &\leq 380 \\
   x_{\text{S5},\text{D1}} + x_{\text{S5},\text{D2}} + x_{\text{S5},\text{D3}} + x_{\text{S5},\text{D4}} + x_{\text{S5},\text{D5}} &\leq 254 \\
   \end{align*}

3. **Nonnegativity:**
   \[
   x_{ij} \geq 0, \quad \forall i, j
   \]

---

**All identifiers, coefficients, and constraints are as retrieved and in source order.**