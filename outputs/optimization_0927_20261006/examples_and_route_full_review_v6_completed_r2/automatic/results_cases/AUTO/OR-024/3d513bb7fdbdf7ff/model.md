Let $x_i$ denote the number of units of product $i$ (with identifier ‘S700_…’) to fulfill.

**Parameters (from data):**

| Product Name   | Revenue ($r_i$) | Demand ($d_i$) | Initial Inventory ($s_i$) |
|----------------|-----------------|----------------|---------------------------|
| S700_1138      | 70.67           | 1219           | 9020                      |
| S700_1691      | 100.0           | 1127           | 8370                      |
| S700_1938      | 70.15           | 1129           | 8390                      |
| S700_2047      | 100.0           | 1176           | 8680                      |
| S700_2466      | 100.0           | 1301           | 9400                      |
| S700_2610      | 65.77           | 1340           | 9900                      |
| S700_2824      | 100.0           | 1357           | 9760                      |
| S700_2834      | 100.0           | 1158           | 8610                      |
| S700_3167      | 74.4            | 1287           | 9380                      |
| S700_3505      | 81.14           | 1281           | 9170                      |
| S700_3962      | 100.0           | 1135           | 8520                      |
| S700_4002      | 61.44           | 1392           | 10290                     |

**Decision variables:**

$x_i \in \mathbb{Z}_{\geq 0}$ for each product $i$ (as listed above)

---

**Objective:**

$$
\max \sum_{i} r_i x_i
$$

---

**Constraints:**

For each product $i$:

1. Demand fulfillment:
$$
x_i \leq d_i
$$

2. Inventory availability:
$$
x_i \leq s_i
$$

3. Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0}
$$

---

**Explicit Model:**

$$
\begin{align*}
\max\quad & 70.67\,x_{\text{S700\_1138}} + 100.0\,x_{\text{S700\_1691}} + 70.15\,x_{\text{S700\_1938}} + 100.0\,x_{\text{S700\_2047}} \\
& + 100.0\,x_{\text{S700\_2466}} + 65.77\,x_{\text{S700\_2610}} + 100.0\,x_{\text{S700\_2824}} + 100.0\,x_{\text{S700\_2834}} \\
& + 74.4\,x_{\text{S700\_3167}} + 81.14\,x_{\text{S700\_3505}} + 100.0\,x_{\text{S700\_3962}} + 61.44\,x_{\text{S700\_4002}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{S700\_1138}} \leq \min\{1219,\,9020\} \\
& 0 \leq x_{\text{S700\_1691}} \leq \min\{1127,\,8370\} \\
& 0 \leq x_{\text{S700\_1938}} \leq \min\{1129,\,8390\} \\
& 0 \leq x_{\text{S700\_2047}} \leq \min\{1176,\,8680\} \\
& 0 \leq x_{\text{S700\_2466}} \leq \min\{1301,\,9400\} \\
& 0 \leq x_{\text{S700\_2610}} \leq \min\{1340,\,9900\} \\
& 0 \leq x_{\text{S700\_2824}} \leq \min\{1357,\,9760\} \\
& 0 \leq x_{\text{S700\_2834}} \leq \min\{1158,\,8610\} \\
& 0 \leq x_{\text{S700\_3167}} \leq \min\{1287,\,9380\} \\
& 0 \leq x_{\text{S700\_3505}} \leq \min\{1281,\,9170\} \\
& 0 \leq x_{\text{S700\_3962}} \leq \min\{1135,\,8520\} \\
& 0 \leq x_{\text{S700\_4002}} \leq \min\{1392,\,10290\} \\
& x_i \in \mathbb{Z}_{\geq 0} \quad \forall i
\end{align*}
$$

where $i$ runs over all products listed above, with their respective coefficients and bounds.