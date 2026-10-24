**Mathematical model**

Let  

* \(P=\{\,\text{Widget1},\dots ,\text{Widget141}\,\}\) – set of widgets.  
* \(x_p\) – number of units of widget \(p\) produced (continuous, \(x_p\ge 0\)).  
* \(s\) – kilograms of Catalyst X sold (continuous, \(s\ge 0\)).  
* \(d\) – kilograms of Catalyst X disposed (continuous, \(d\ge 0\)).  

Parameters (all taken directly from the supplied files)

| Symbol | Source file | Column | Meaning |
|--------|-------------|--------|---------|
| \(L_p\) | `product_resources.csv` | `LaborHours` | labor hours per unit of widget \(p\) |
| \(A_p\) | `product_resources.csv` | `MaterialA` | kg of Material A per unit of widget \(p\) |
| \(B_p\) | `product_resources.csv` | `MaterialB` | kg of Material B per unit of widget \(p\) |
| \(\pi_p\) | `product_resources.csv` | `Profit` | profit per unit of widget \(p\) |
| \(R_L\) | `resource_limits.csv` | `MonthlyLimit` for `LaborHours` | 5 000 |
| \(R_A\) | `resource_limits.csv` | `MonthlyLimit` for `MaterialA` | 24 000 |
| \(R_B\) | `resource_limits.csv` | `MonthlyLimit` for `MaterialB` | 15 000 |
| \(c_{\text{sell}}\) | – | – | 300 (price per kg of Catalyst X sold) |
| \(c_{\text{dispose}}\) | – | – | 200 (disposal cost per kg of Catalyst X) |
| \(S_{\max}\) | – | – | 1 500 (maximum kg of Catalyst X that can be sold) |
| \(k\) | – | – | 5 (kg of Catalyst X produced per unit of Widget 3) |

**Objective**

\[
\max \;\sum_{p\in P}\pi_p\,x_p \;+\; c_{\text{sell}}\;s \;-\; c_{\text{dispose}}\;d
\]

**Constraints**

1. **Resource limits**

\[
\sum_{p\in P} L_p\,x_p \;\le\; R_L
\]
\[
\sum_{p\in P} A_p\,x_p \;\le\; R_A
\]
\[
\sum_{p\in P} B_p\,x_p \;\le\; R_B
\]

2. **Catalyst X balance**

\[
k\,x_{\text{Widget3}} \;=\; s + d
\]

3. **Sales cap**

\[
s \;\le\; S_{\max}
\]

4. **Non‑negativity**

\[
x_p \ge 0 \;\;\forall p\in P,\qquad s\ge 0,\qquad d\ge 0
\]

---

### Data‑Mapping rules

| Variable / Parameter | Source file | Column | Mapping rule |
|----------------------|-------------|--------|--------------|
| \(x_p\) | `product_resources.csv` | `Product` | index \(p\) is the product name |
| \(L_p\) | `product_resources.csv` | `LaborHours` | value for product \(p\) |
| \(A_p\) | `product_resources.csv` | `MaterialA` | value for product \(p\) |
| \(B_p\) | `product_resources.csv` | `MaterialB` | value for product \(p\) |
| \(\pi_p\) | `product_resources.csv` | `Profit` | value for product \(p\) |
| \(R_L\) | `resource_limits.csv` | `MonthlyLimit` where `Resource`=`LaborHours` | 5000 |
| \(R_A\) | `resource_limits.csv` | `MonthlyLimit` where `Resource`=`MaterialA` | 24000 |
| \(R_B\) | `resource_limits.csv` | `MonthlyLimit` where `Resource`=`MaterialB` | 15000 |
| \(k\) | – | – | constant 5 (given in problem text) |
| \(c_{\text{sell}}\) | – | – | constant 300 |
| \(c_{\text{dispose}}\) | – | – | constant 200 |
| \(S_{\max}\) | – | – | constant 1500 |
| \(s\) | – | – | decision variable for sold Catalyst X |
| \(d\) | – | – | decision variable for disposed Catalyst X |

All other constants (objective coefficients, resource limits, and the Catalyst X production rate) are taken verbatim from the problem statement and the supplied CSV files. No additional entities or constraints are introduced beyond those explicitly described.