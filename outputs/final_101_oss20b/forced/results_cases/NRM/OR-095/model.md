**Mathematical Model**

Let  

* \(I=\{p\mid p\text{ is a product in } \text{product\_resources.csv}\}\)  
  (i.e. \(I=\{\text{Widget1},\dots ,\text{Widget141}\}\))  

* \(x_p\) – number of units of product \(p\) produced (integer, \(\ge 0\))  

* \(y\) – kilograms of CatalystX sold (continuous, \(\ge 0\))

Parameters (all taken from the data tables unless noted otherwise)

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(L_p\) | Labor hours per unit of product \(p\) | `file_1_view_0` – column **LaborHours** |
| \(A_p\) | Material A consumption per unit of product \(p\) | `file_1_view_0` – column **MaterialA** |
| \(B_p\) | Material B consumption per unit of product \(p\) | `file_1_view_0` – column **MaterialB** |
| \(\pi_p\) | Base profit per unit of product \(p\) | `file_1_view_0` – column **Profit** |
| \(L^{\max}\) | Monthly labor‑hour limit | `file_0_view_0` – row where **Resource**=`LaborHours` |
| \(A^{\max}\) | Monthly Material A limit | `file_0_view_0` – row where **Resource**=`MaterialA` |
| \(B^{\max}\) | Monthly Material B limit | `file_0_view_0` – row where **Resource**=`MaterialB` |
| \(r\) | CatalystX produced per unit of Widget3 | 5 kg (given) |
| \(p_{\text{sell}}\) | Sale price of CatalystX | 300 $/kg (given) |
| \(c_{\text{dispose}}\) | Disposal cost of unsold CatalystX | 200 $/kg (given) |
| \(y^{\max}\) | Market cap for CatalystX sales | 1500 kg (given) |

---

### Objective

\[
\max \;\; \sum_{p\in I}\pi_p\,x_p \;-\; 1000\,x_{\text{Widget3}} \;+\; 500\,y
\]

*(The terms \(-1000\,x_{\text{Widget3}}\) and \(+500\,y\) arise from the sale price
\(p_{\text{sell}}\) and disposal cost \(c_{\text{dispose}}\) applied to the
catalyst produced by Widget3.)*

---

### Constraints

1. **Labor capacity**

\[
\sum_{p\in I} L_p\,x_p \;\le\; L^{\max}
\]

2. **Material A capacity**

\[
\sum_{p\in I} A_p\,x_p \;\le\; A^{\max}
\]

3. **Material B capacity**

\[
\sum_{p\in I} B_p\,x_p \;\le\; B^{\max}
\]

4. **CatalystX production and sale**

\[
\begin{aligned}
&0 \;\le\; y \;\le\; y^{\max} \\
&y \;\le\; r\,x_{\text{Widget3}}
\end{aligned}
\]

5. **Variable domains**

\[
x_p \in \mathbb{Z}_{+}\quad \forall p\in I,\qquad
y \in \mathbb{R}_{+}
\]

---

**Data Mapping**

| Parameter | Table ID | Column Name |
|-----------|----------|-------------|
| \(L_p\) | `file_1_view_0` | `LaborHours` |
| \(A_p\) | `file_1_view_0` | `MaterialA` |
| \(B_p\) | `file_1_view_0` | `MaterialB` |
| \(\pi_p\) | `file_1_view_0` | `Profit` |
| \(L^{\max}\) | `file_0_view_0` | `MonthlyLimit` (where `Resource`=`LaborHours`) |
| \(A^{\max}\) | `file_0_view_0` | `MonthlyLimit` (where `Resource`=`MaterialA`) |
| \(B^{\max}\) | `file_0_view_0` | `MonthlyLimit` (where `Resource`=`MaterialB`) |

All other constants (\(r\), \(p_{\text{sell}}\), \(c_{\text{dispose}}\), \(y^{\max}\)) are fixed values supplied in the problem statement.