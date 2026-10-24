**Mathematical Model**

Let  

* \(I=\{i\mid \text{Product}= \text{Widget}k,\;k=1,\dots ,141\}\)  
* \(x_i\) – number of units of widget \(i\) produced (integer, \(\ge 0\))  
* \(s\) – kilograms of Catalyst X sold (continuous, \(\ge 0\))  
* \(u\) – kilograms of Catalyst X disposed (continuous, \(\ge 0\))

Parameters (see *Data Mapping* below)

* \(\text{Labor}_i,\;\text{MatA}_i,\;\text{MatB}_i,\;\text{Profit}_i\) – from **product_resources.csv**  
* \(L_{\text{Labor}},\;L_{\text{MatA}},\;L_{\text{MatB}}\) – from **resource_limits.csv**  
* \(c_{\text{sell}}=300\) – sale price of Catalyst X (kg)  
* \(c_{\text{dispose}}=200\) – disposal cost of Catalyst X (kg)  
* \(g=5\) – kg of Catalyst X generated per unit of **Widget3**

---

\[
\begin{aligned}
\max\;&\sum_{i\in I}\text{Profit}_i\,x_i \;+\; c_{\text{sell}}\,s \;-\; c_{\text{dispose}}\,u \\[4pt]
\text{s.t. }&
\sum_{i\in I}\text{Labor}_i\,x_i \;\le\; L_{\text{Labor}} \quad\text{(Labor)}\\
&\sum_{i\in I}\text{MatA}_i\,x_i \;\le\; L_{\text{MatA}} \quad\text{(Material A)}\\
&\sum_{i\in I}\text{MatB}_i\,x_i \;\le\; L_{\text{MatB}} \quad\text{(Material B)}\\[4pt]
&u \;=\; g\,x_{\text{Widget3}} \;-\; s \quad\text{(Catalyst balance)}\\
&0 \;\le\; s \;\le\; 1500 \quad\text{(Market cap)}\\
&u \;\ge\; 0 \quad\text{(Non‑negativity)}\\
&x_i \;\in\;\mathbb{Z}_{\ge 0}\quad\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(L_{\text{Labor}}\) | `file_0_view_0` | `MonthlyLimit` | row where `Resource = 'LaborHours'` |
| \(L_{\text{MatA}}\) | `file_0_view_0` | `MonthlyLimit` | row where `Resource = 'MaterialA'` |
| \(L_{\text{MatB}}\) | `file_0_view_0` | `MonthlyLimit` | row where `Resource = 'MaterialB'` |
| \(\text{Labor}_i\) | `file_1_view_0` | `LaborHours` | for each `Product` |
| \(\text{MatA}_i\) | `file_1_view_0` | `MaterialA` | for each `Product` |
| \(\text{MatB}_i\) | `file_1_view_0` | `MaterialB` | for each `Product` |
| \(\text{Profit}_i\) | `file_1_view_0` | `Profit` | for each `Product` |
| \(x_{\text{Widget3}}\) | – | – | index \(i\) where `Product = 'Widget3'` |
| \(c_{\text{sell}}\) | – | – | constant 300 |
| \(c_{\text{dispose}}\) | – | – | constant 200 |
| \(g\) | – | – | constant 5 |

All other constants (sale cap 1500 kg, integrality of \(x_i\), non‑negativity of \(s,u\)) are defined in the model formulation.