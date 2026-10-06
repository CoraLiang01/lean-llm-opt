##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$ (continuous).

##### Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Parameters

- Demand at outlet $j$: $d_j$
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$
- Supply capacity at plant $i$: $s_i$
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$
- Transportation cost per unit from $i$ to $j$: $c_{ij}$

|        | C1                | C2                | C3                | C4                |
|--------|-------------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856  | 23.685276141764653| 23.676386730773032| 447.75143678673766|
| S2     | 883.9151090405642 | 0.04977684765576961| 0.0350986687216299| 44.45588531711622 |
| S3     | 537.3456896658107 | 23.769274659075112| 498.95659249465467| 440.60737890439776|
| S4     | 1791.493192397229 | 68.21633865655126 | 1432.4837339656747| 1527.7635425462734|

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each retail outlet $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   - $\sum_{i} x_{i,\text{C1}} \geq 94$
   - $\sum_{i} x_{i,\text{C2}} \geq 39$
   - $\sum_{i} x_{i,\text{C3}} \geq 65$
   - $\sum_{i} x_{i,\text{C4}} \geq 435$

2. **Supply capacity:**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   - $\sum_{j} x_{\text{S1},j} \leq 2531$
   - $\sum_{j} x_{\text{S2},j} \leq 20$
   - $\sum_{j} x_{\text{S3},j} \leq 210$
   - $\sum_{j} x_{\text{S4},j} \leq 241$

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- customer_demand.csv:  
  - customer $\to$ $j \in J$
  - demand $\to$ $d_j$
- supply_capacity.csv:  
  - Unnamed: 0 $\to$ $i \in I$
  - supply_capacity $\to$ $s_i$
- transportation_costs.csv:  
  - Unnamed: 0 $\to$ $i \in I$
  - $C1$, $C2$, $C3$, $C4$ $\to$ $c_{ij}$ for $j \in J$