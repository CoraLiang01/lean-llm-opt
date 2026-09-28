Let $x_{ij}$ be the quantity shipped from supplier $i$ to customer $j$.

#### Sets and Indices
- Suppliers $i \in \{\text{supply1}, \text{supply2}, \text{supply3}, \text{supply4}, \text{supply5}, \text{supply6}, \text{supply7}, \text{supply8}\}$
- Customers $j \in \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

#### Parameters

- Demand for each customer:
  - demand1: $9$
  - demand2: $66$
  - demand3: $56$
  - demand4: $17$
  - demand5: $43$
  - demand6: $62$
  - demand7: $10$
  - demand8: $37$

- Supply capacity for each supplier:
  - supplier1: $60$
  - supplier2: $22$
  - supplier3: $16$
  - supplier4: $14$
  - supplier5: $19$
  - supplier6: $70$
  - supplier7: $60$
  - supplier8: $39$

- Transportation costs $c_{ij}$ (cost per unit from supplier $i$ to customer $j$):

|           | demand1      | demand2      | demand3      | demand4      | demand5      | demand6      | demand7      | demand8      |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|
| supply1   | 0.0302073664 | 229.50723505 | 198.62356558 | 12.99505064  | 211.20732124 | 134.94429850 | 9.822206399  | 11.39407754  |
| supply2   | 232.34691308 | 3.625872644  | 0.2860543415 | 45.73127693  | 2.830479656  | 107.05891033 | 299.96317913 | 23.79935436  |
| supply3   | 11.06193833  | 0.204199533  | 0.278944728  | 45.72191272  | 59.54895566  | 5.097536740  | 300.00118415 | 23.71128271  |
| supply4   | 235.17948357 | 43.79466896  | 40.70984678  | 0.0777449662 | 4.237728183  | 131.70915517 | 296.55587568 | 29.81094002  |
| supply5   | 211.85808746 | 47.60180877  | 50.04007716  | 86.14548807  | 0.0619789792 | 5.334551530  | 270.06290424 | 3.853933134  |
| supply6   | 6.455066336  | 88.16323623  | 5.047091672  | 151.46120287 | 5.290760161  | 0.0460220534 | 9.936706602  | 103.75460989 |
| supply7   | 174.27229047 | 250.58223529 | 253.90413042 | 16.23546732  | 12.64314051  | 175.06728241 | 2.983839625  | 317.06551939 |
| supply8   | 207.87006254 | 1.517168472  | 24.02723929  | 27.13399928  | 73.20672469  | 125.72910360 | 15.46310325  | 0.2016498751 |

#### Decision Variables

- $x_{ij} \geq 0$ (continuous), for all suppliers $i$ and customers $j$

#### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in \{\text{supply1},\ldots,\text{supply8}\}} \sum_{j \in \{\text{demand1},\ldots,\text{demand8}\}} c_{ij} x_{ij}
$$

#### Constraints

1. **Demand satisfaction (for each customer $j$):**
   $$
   \sum_{i} x_{ij} = d_j \qquad \forall j \in \{\text{demand1},\ldots,\text{demand8}\}
   $$
   where $d_j$ is the demand for customer $j$.

   - demand1: $\sum_{i} x_{i,\text{demand1}} = 9$
   - demand2: $\sum_{i} x_{i,\text{demand2}} = 66$
   - demand3: $\sum_{i} x_{i,\text{demand3}} = 56$
   - demand4: $\sum_{i} x_{i,\text{demand4}} = 17$
   - demand5: $\sum_{i} x_{i,\text{demand5}} = 43$
   - demand6: $\sum_{i} x_{i,\text{demand6}} = 62$
   - demand7: $\sum_{i} x_{i,\text{demand7}} = 10$
   - demand8: $\sum_{i} x_{i,\text{demand8}} = 37$

2. **Supply capacity (for each supplier $i$):**
   $$
   \sum_{j} x_{ij} \leq s_i \qquad \forall i \in \{\text{supply1},\ldots,\text{supply8}\}
   $$
   where $s_i$ is the supply capacity for supplier $i$.

   - supply1: $\sum_{j} x_{\text{supply1},j} \leq 60$
   - supply2: $\sum_{j} x_{\text{supply2},j} \leq 22$
   - supply3: $\sum_{j} x_{\text{supply3},j} \leq 16$
   - supply4: $\sum_{j} x_{\text{supply4},j} \leq 14$
   - supply5: $\sum_{j} x_{\text{supply5},j} \leq 19$
   - supply6: $\sum_{j} x_{\text{supply6},j} \leq 70$
   - supply7: $\sum_{j} x_{\text{supply7},j} \leq 60$
   - supply8: $\sum_{j} x_{\text{supply8},j} \leq 39$

3. **Nonnegativity:**
   $$
   x_{ij} \geq 0 \qquad \forall i, j
   $$

---

**Complete Model:**

Minimize
$$
\sum_{i \in \{\text{supply1},\ldots,\text{supply8}\}} \sum_{j \in \{\text{demand1},\ldots,\text{demand8}\}} c_{ij} x_{ij}
$$

Subject to:
$$
\sum_{i} x_{ij} = d_j \qquad \forall j \in \{\text{demand1},\ldots,\text{demand8}\}
$$
$$
\sum_{j} x_{ij} \leq s_i \qquad \forall i \in \{\text{supply1},\ldots,\text{supply8}\}
$$
$$
x_{ij} \geq 0 \qquad \forall i, j
$$

Where all parameters and indices are as specified above.