Let $x_{ij}$ denote the number of units shipped from warehouse (region) $i$ to store (customer) $j$, where $i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ and $j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$.

**Parameters:**

- Demand for each store:
  - D1: $428$
  - D2: $217$
  - D3: $214$
  - D4: $380$
  - D5: $254$

- Supply capacity for each warehouse:
  - S1: $428$
  - S2: $217$
  - S3: $214$
  - S4: $380$
  - S5: $254$

- Transportation costs per unit from each warehouse to each store:

|        | D1                | D2                | D3                | D4                | D5                |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|
| S1     | 269.3910588020795 | 1.4537335390933939| 99.60345345756605 | 26.64078166309837 | 9.537688956880922 |
| S2     | 9.291846876785183 | 10.874778437070223| 144.52609291614627| 11.420133077898234| 153.1756819927813 |
| S3     | 9.674584301671008 | 2.6191650959687944| 100.8242249168735 | 3.2121910887916876| 133.8493396124168 |
| S4     | 270.57498480010247| 32.50253586       | 4.6842098096469815| 1.5682269686546804| 9.58927599        |
| S5     | 226.0331910675782 | 8.669161980826471 | 65.47681316968448 | 9.068765258459958 | 202.65015316425533|

---

**Mathematical Model:**

**Objective:**
\[
\min \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}} \sum_{j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}} c_{ij} \, x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from warehouse $i$ to store $j$ as given above.

**Subject to:**

1. **Supply capacity at each warehouse:**
   \[
   \sum_{j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}} x_{ij} \leq \text{supply\_capacity}_i \qquad \forall i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}
   \]
   - S1: $\sum_j x_{\text{S1},j} \leq 428$
   - S2: $\sum_j x_{\text{S2},j} \leq 217$
   - S3: $\sum_j x_{\text{S3},j} \leq 214$
   - S4: $\sum_j x_{\text{S4},j} \leq 380$
   - S5: $\sum_j x_{\text{S5},j} \leq 254$

2. **Demand satisfaction at each store:**
   \[
   \sum_{i \in \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}} x_{ij} = \text{demand}_j \qquad \forall j \in \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}
   \]
   - D1: $\sum_i x_{i,\text{D1}} = 428$
   - D2: $\sum_i x_{i,\text{D2}} = 217$
   - D3: $\sum_i x_{i,\text{D3}} = 214$
   - D4: $\sum_i x_{i,\text{D4}} = 380$
   - D5: $\sum_i x_{i,\text{D5}} = 254$

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i, j
   \]

---

**Decision Variables:**

- $x_{ij}$: Number of units shipped from warehouse $i$ to store $j$ (continuous, $x_{ij} \geq 0$).

---

**All data and identifiers are preserved in source order as required.**