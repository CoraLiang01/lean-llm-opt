**Sets:**

- Warehouses $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- Stores $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

**Parameters:**

- Demand for each store:
  - $d_{\text{D1}} = 428$
  - $d_{\text{D2}} = 217$
  - $d_{\text{D3}} = 214$
  - $d_{\text{D4}} = 380$
  - $d_{\text{D5}} = 254$
- Supply capacity for each warehouse:
  - $s_{\text{S1}} = 428$
  - $s_{\text{S2}} = 217$
  - $s_{\text{S3}} = 214$
  - $s_{\text{S4}} = 380$
  - $s_{\text{S5}} = 254$
- Transportation costs $c_{ij}$ (from warehouse $i$ to store $j$):

|         | D1                | D2                | D3                | D4                | D5                |
|---------|-------------------|-------------------|-------------------|-------------------|-------------------|
| S1      | 269.3910588020795 | 1.4537335390933939| 99.60345345756605 | 26.64078166309837 | 9.537688956880922 |
| S2      | 9.291846876785183 | 10.874778437070223| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
| S3      | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.2121910887916876| 133.8493396124168 |
| S4      | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
| S5      | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|

**Decision Variables:**

- $x_{ij} \geq 0$ (continuous): quantity shipped from warehouse $i$ to store $j$

**Objective:**

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

**Constraints:**

1. **Demand satisfaction (each store):**
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
   Explicitly:
   - $x_{\text{S1},\text{D1}} + x_{\text{S2},\text{D1}} + x_{\text{S3},\text{D1}} + x_{\text{S4},\text{D1}} + x_{\text{S5},\text{D1}} \geq 428$
   - $x_{\text{S1},\text{D2}} + x_{\text{S2},\text{D2}} + x_{\text{S3},\text{D2}} + x_{\text{S4},\text{D2}} + x_{\text{S5},\text{D2}} \geq 217$
   - $x_{\text{S1},\text{D3}} + x_{\text{S2},\text{D3}} + x_{\text{S3},\text{D3}} + x_{\text{S4},\text{D3}} + x_{\text{S5},\text{D3}} \geq 214$
   - $x_{\text{S1},\text{D4}} + x_{\text{S2},\text{D4}} + x_{\text{S3},\text{D4}} + x_{\text{S4},\text{D4}} + x_{\text{S5},\text{D4}} \geq 380$
   - $x_{\text{S1},\text{D5}} + x_{\text{S2},\text{D5}} + x_{\text{S3},\text{D5}} + x_{\text{S4},\text{D5}} + x_{\text{S5},\text{D5}} \geq 254$

2. **Supply capacity (each warehouse):**
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
   Explicitly:
   - $x_{\text{S1},\text{D1}} + x_{\text{S1},\text{D2}} + x_{\text{S1},\text{D3}} + x_{\text{S1},\text{D4}} + x_{\text{S1},\text{D5}} \leq 428$
   - $x_{\text{S2},\text{D1}} + x_{\text{S2},\text{D2}} + x_{\text{S2},\text{D3}} + x_{\text{S2},\text{D4}} + x_{\text{S2},\text{D5}} \leq 217$
   - $x_{\text{S3},\text{D1}} + x_{\text{S3},\text{D2}} + x_{\text{S3},\text{D3}} + x_{\text{S3},\text{D4}} + x_{\text{S3},\text{D5}} \leq 214$
   - $x_{\text{S4},\text{D1}} + x_{\text{S4},\text{D2}} + x_{\text{S4},\text{D3}} + x_{\text{S4},\text{D4}} + x_{\text{S4},\text{D5}} \leq 380$
   - $x_{\text{S5},\text{D1}} + x_{\text{S5},\text{D2}} + x_{\text{S5},\text{D3}} + x_{\text{S5},\text{D4}} + x_{\text{S5},\text{D5}} \leq 254$

3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

**Full Model:**

Minimize
$$
\begin{align*}
&269.3910588020795\, x_{\text{S1},\text{D1}} + 1.4537335390933939\, x_{\text{S1},\text{D2}} + 99.60345345756605\, x_{\text{S1},\text{D3}} + 26.64078166309837\, x_{\text{S1},\text{D4}} + 9.537688956880922\, x_{\text{S1},\text{D5}} \\
+&9.291846876785183\, x_{\text{S2},\text{D1}} + 10.874778437070223\, x_{\text{S2},\text{D2}} + 144.52609291614627\, x_{\text{S2},\text{D3}} + 11.420133077898234\, x_{\text{S2},\text{D4}} + 153.1756819927813\, x_{\text{S2},\text{D5}} \\
+&9.674584301671008\, x_{\text{S3},\text{D1}} + 2.6191650959687944\, x_{\text{S3},\text{D2}} + 100.8242249168735\, x_{\text{S3},\text{D3}} + 3.2121910887916876\, x_{\text{S3},\text{D4}} + 133.8493396124168\, x_{\text{S3},\text{D5}} \\
+&270.57498480010247\, x_{\text{S4},\text{D1}} + 32.50253586\, x_{\text{S4},\text{D2}} + 4.6842098096469815\, x_{\text{S4},\text{D3}} + 1.5682269686546804\, x_{\text{S4},\text{D4}} + 9.58927599\, x_{\text{S4},\text{D5}} \\
+&226.0331910675782\, x_{\text{S5},\text{D1}} + 8.669161980826471\, x_{\text{S5},\text{D2}} + 65.47681316968448\, x_{\text{S5},\text{D3}} + 9.068765258459958\, x_{\text{S5},\text{D4}} + 202.65015316425533\, x_{\text{S5},\text{D5}}
\end{align*}
$$

Subject to:

For each store:
- $x_{\text{S1},j} + x_{\text{S2},j} + x_{\text{S3},j} + x_{\text{S4},j} + x_{\text{S5},j} \geq d_j$ for $j = \text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}$

For each warehouse:
- $x_{i,\text{D1}} + x_{i,\text{D2}} + x_{i,\text{D3}} + x_{i,\text{D4}} + x_{i,\text{D5}} \leq s_i$ for $i = \text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}$

And
- $x_{ij} \geq 0$ for all $i \in I$, $j \in J$