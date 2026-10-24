**Mathematical Model**

Let  

* \(P=\{\,\text{Widget1},\dots ,\text{Widget141}\,\}\) – set of products.  
* \(x_p\ge 0\) – units of product \(p\in P\) produced in the month.  
* \(s\ge 0\) – kilograms of Catalyst X sold in the month.

Parameters (all obtained from the supplied CSV files)

| Symbol | Meaning | Source table & column |
|--------|---------|-----------------------|
| \(L_p\) | Labor hours per unit of product \(p\) | `product_resources.csv` – `LaborHours` |
| \(A_p\) | Material A consumption per unit of product \(p\) | `product_resources.csv` – `MaterialA` |
| \(B_p\) | Material B consumption per unit of product \(p\) | `product_resources.csv` – `MaterialB` |
| \(\pi_p\) | Base profit per unit of product \(p\) | `product_resources.csv` – `Profit` |
| \(R_L\) | Monthly labor‑hour limit | `resource_limits.csv` – `MonthlyLimit` where `Resource='LaborHours'` |
| \(R_A\) | Monthly Material A limit | `resource_limits.csv` – `MonthlyLimit` where `Resource='MaterialA'` |
| \(R_B\) | Monthly Material B limit | `resource_limits.csv` – `MonthlyLimit` where `Resource='MaterialB'` |
| \(g\) | Catalyst X generated per unit of **Widget3** | constant \(g=5\) kg |
| \(p_{\text{sell}}\) | Sale price of Catalyst X | constant \(p_{\text{sell}}=300\) $/kg |
| \(p_{\text{disp}}\) | Disposal cost of unsold Catalyst X | constant \(p_{\text{disp}}=200\) $/kg |
| \(S_{\max}\) | Monthly sales cap for Catalyst X | constant \(S_{\max}=1500\) kg |

---

### Objective

\[
\max \;\; \sum_{p\in P}\pi_p\,x_p \;-\; g\,p_{\text{disp}}\;x_{\text{Widget3}}
\;+\; \bigl(p_{\text{sell}}+p_{\text{disp}}\bigr)\,s
\]

(Equivalently \( \max \sum_{p}\pi_p x_p -1000\,x_{\text{Widget3}} + 500\,s\).)

---

### Constraints

1. **Catalyst X sales limits**

\[
\begin{aligned}
s &\le S_{\max} \\
s &\le g\,x_{\text{Widget3}}
\end{aligned}
\]

2. **Resource limits**

\[
\begin{aligned}
\sum_{p\in P} L_p\,x_p &\le R_L \\
\sum_{p\in P} A_p\,x_p &\le R_A \\
\sum_{p\in P} B_p\,x_p &\le R_B
\end{aligned}
\]

3. **Non‑negativity**

\[
x_p \ge 0 \;\;\forall p\in P,\qquad s \ge 0
\]

---

**Data Mapping Rules**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(L_p\) | `file_1_view_0` | `LaborHours` |
| \(A_p\) | `file_1_view_0` | `MaterialA` |
| \(B_p\) | `file_1_view_0` | `MaterialB` |
| \(\pi_p\) | `file_1_view_0` | `Profit` |
| \(R_L\) | `file_0_view_0` | `MonthlyLimit` where `Resource='LaborHours'` |
| \(R_A\) | `file_0_view_0` | `MonthlyLimit` where `Resource='MaterialA'` |
| \(R_B\) | `file_0_view_0` | `MonthlyLimit` where `Resource='MaterialB'` |
| \(g\) | – | constant 5 |
| \(p_{\text{sell}}\) | – | constant 300 |
| \(p_{\text{disp}}\) | – | constant 200 |
| \(S_{\max}\) | – | constant 1500 |

All other constants are defined in the model description. No additional data or aggregation is required.