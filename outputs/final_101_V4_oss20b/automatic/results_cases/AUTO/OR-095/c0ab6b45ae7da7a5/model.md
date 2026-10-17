## Mathematical Model

**Index set**  
\(I=\{\text{Widget1},\text{Widget2},\dots ,\text{Widget141}\}\)

**Parameters**  

\[
\begin{aligned}
a_i &\;=\;\text{LaborHours per unit of product }i\\
b_i &\;=\;\text{MaterialA per unit of product }i\\
c_i &\;=\;\text{MaterialB per unit of product }i\\
p_i &\;=\;\text{Base profit per unit of product }i\\
r   &\;=\;5 \quad\text{(kg CatalystX produced per unit of Widget3)}\\
s_{\max} &\;=\;1500 \quad\text{(kg CatalystX that can be sold)}\\
\pi_s &\;=\;300 \quad\text{($/kg$ sold)}\\
\delta &\;=\;200 \quad\text{($/kg$ disposal)}\\
L_{\text{lab}} &\;=\;5000 \quad\text{(monthly labor hours)}\\
L_{\text{A}}   &\;=\;24000 \quad\text{(kg MaterialA)}\\
L_{\text{B}}   &\;=\;15000 \quad\text{(kg MaterialB)}
\end{aligned}
\]

**Decision variables**  

\[
\begin{aligned}
x_i &\ge 0 \quad\text{(units of product }i\text{ produced)}\\
s   &\ge 0 \quad\text{(kg of CatalystX sold)}
\end{aligned}
\]

**Objective**  

\[
\max \;\sum_{i\in I} p_i\,x_i \;+\; \pi_s\,s \;-\; \delta\,(r\,x_{\text{Widget3}}-s)
\]

(Equivalently \( \max \sum_i p_i x_i + (\pi_s+\delta)s - \delta r\,x_{\text{Widget3}}\).)

**Constraints**

\[
\begin{aligned}
\sum_{i\in I} a_i\,x_i &\le L_{\text{lab}} &&\text{(labor)}\\
\sum_{i\in I} b_i\,x_i &\le L_{\text{A}}   &&\text{(Material A)}\\
\sum_{i\in I} c_i\,x_i &\le L_{\text{B}}   &&\text{(Material B)}\\
s &\le s_{\max} &&\text{(sales cap)}\\
s &\le r\,x_{\text{Widget3}} &&\text{(cannot sell more than produced)}\\
x_i &\ge 0,\; s\ge 0 &&\forall i\in I
\end{aligned}
\]

---

## Data Mapping Rules

| Parameter | Source | Table ID | Column | Mapping Rule |
|-----------|--------|----------|--------|--------------|
| \(a_i\) | Labor hours per unit | `file_1_view_0` | `LaborHours` | For each product \(i\in I\), \(a_i\) = value in `LaborHours` where `Product` = \(i\). |
| \(b_i\) | Material A per unit | `file_1_view_0` | `MaterialA` | For each product \(i\in I\), \(b_i\) = value in `MaterialA` where `Product` = \(i\). |
| \(c_i\) | Material B per unit | `file_1_view_0` | `MaterialB` | For each product \(i\in I\), \(c_i\) = value in `MaterialB` where `Product` = \(i\). |
| \(p_i\) | Base profit per unit | `file_1_view_0` | `Profit` | For each product \(i\in I\), \(p_i\) = value in `Profit` where `Product` = \(i\). |
| \(r\) | CatalystX per unit of Widget3 | constant | – | \(r = 5\) (given in problem statement). |
| \(s_{\max}\) | Sales cap | constant | – | \(s_{\max} = 1500\). |
| \(\pi_s\) | Sale price of CatalystX | constant | – | \(\pi_s = 300\). |
| \(\delta\) | Disposal cost per kg | constant | – | \(\delta = 200\). |
| \(L_{\text{lab}}\) | Labor limit | `file_0_view_0` | `MonthlyLimit` | \(L_{\text{lab}} = 5000\) where `Resource` = `LaborHours`. |
| \(L_{\text{A}}\) | Material A limit | `file_0_view_0` | `MonthlyLimit` | \(L_{\text{A}} = 24000\) where `Resource` = `MaterialA`. |
| \(L_{\text{B}}\) | Material B limit | `file_0_view_0` | `MonthlyLimit` | \(L_{\text{B}} = 15000\) where `Resource` = `MaterialB`. |

All parameters are taken directly from the provided CSV files; no additional rows or values are invented. The model uses continuous, non‑negative production quantities and CatalystX sales, consistent with the problem statement.