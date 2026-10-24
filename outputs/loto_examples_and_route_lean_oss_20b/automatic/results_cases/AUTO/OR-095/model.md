**Mathematical Model**

Let  

* \(P\) be the set of all products (Widget1 … Widget141).  
* \(x_p\) – number of units of product \(p\in P\) produced (integer, \(\ge 0\)).  
* \(s\) – kilograms of CatalystX sold (continuous, \(\ge 0\)).  
* \(d\) – kilograms of CatalystX disposed (continuous, \(\ge 0\)).

Parameters (all obtained from the supplied CSV tables)

| Symbol | Description | Source |
|--------|-------------|--------|
| \(\text{LH}_p\) | Labor hours per unit of product \(p\) | `file_1_view_0`, column `LaborHours` |
| \(\text{MA}_p\) | Material A consumption per unit of product \(p\) | `file_1_view_0`, column `MaterialA` |
| \(\text{MB}_p\) | Material B consumption per unit of product \(p\) | `file_1_view_0`, column `MaterialB` |
| \(\text{Prof}_p\) | Base profit per unit of product \(p\) | `file_1_view_0`, column `Profit` |
| \(\text{LH}^{\max}\) | Monthly labor‑hour limit | `file_0_view_0`, row where `Resource="LaborHours"` |
| \(\text{MA}^{\max}\) | Monthly Material A limit | `file_0_view_0`, row where `Resource="MaterialA"` |
| \(\text{MB}^{\max}\) | Monthly Material B limit | `file_0_view_0`, row where `Resource="MaterialB"` |
| \(\text{CProd}\) | CatalystX produced per unit of Widget3 | constant \(5\) |
| \(\text{CPrice}\) | Sale price of CatalystX | constant \(300\) |
| \(\text{CDisc}\) | Disposal cost of CatalystX | constant \(200\) |
| \(\text{CCap}\) | Monthly sales cap for CatalystX | constant \(1500\) |

---

### Objective
\[
\max \;\; \sum_{p\in P} \text{Prof}_p\,x_p \;+\; \text{CPrice}\,s \;-\; \text{CDisc}\,d
\]

### Constraints

1. **Labor capacity**
   \[
   \sum_{p\in P} \text{LH}_p\,x_p \;\le\; \text{LH}^{\max}
   \]

2. **Material A capacity**
   \[
   \sum_{p\in P} \text{MA}_p\,x_p \;\le\; \text{MA}^{\max}
   \]

3. **Material B capacity**
   \[
   \sum_{p\in P} \text{MB}_p\,x_p \;\le\; \text{MB}^{\max}
   \]

4. **CatalystX production–sale–disposal balance**
   \[
   s + d \;=\; \text{CProd}\;x_{\text{Widget3}}
   \]

5. **CatalystX sales cap**
   \[
   s \;\le\; \text{CCap}
   \]

6. **Non‑negativity and integrality**
   \[
   x_p \in \mathbb{Z}_{\ge 0}\quad \forall p\in P,\qquad
   s,d \in \mathbb{R}_{\ge 0}
   \]

---

**Data Mapping Rules**

| Parameter | Table ID | Column | Filter (if any) |
|-----------|----------|--------|-----------------|
| \(\text{LH}_p\) | `file_1_view_0` | `LaborHours` | – |
| \(\text{MA}_p\) | `file_1_view_0` | `MaterialA` | – |
| \(\text{MB}_p\) | `file_1_view_0` | `MaterialB` | – |
| \(\text{Prof}_p\) | `file_1_view_0` | `Profit` | – |
| \(\text{LH}^{\max}\) | `file_0_view_0` | `MonthlyLimit` | `Resource = "LaborHours"` |
| \(\text{MA}^{\max}\) | `file_0_view_0` | `MonthlyLimit` | `Resource = "MaterialA"` |
| \(\text{MB}^{\max}\) | `file_0_view_0` | `MonthlyLimit` | `Resource = "MaterialB"` |
| \(\text{CProd}\) | – | – | constant 5 |
| \(\text{CPrice}\) | – | – | constant 300 |
| \(\text{CDisc}\) | – | – | constant 200 |
| \(\text{CCap}\) | – | – | constant 1500 |

All other data (product identifiers, indices) are taken directly from the `Product` column of `file_1_view_0`. The model uses the full set of 141 products and the specific production rate for Widget3 as specified.