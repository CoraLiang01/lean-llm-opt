**Mathematical Model**

Let  

* \(P=\{\,\text{Widget1},\dots ,\text{Widget141}\,\}\) – set of products.  
* \(x_p\) – number of units of product \(p\in P\) produced in the month.  
* \(s\) – kilograms of CatalystX sold to pharmaceutical partners.

All variables are continuous and non‑negative.

\[
\begin{aligned}
\max\;&\sum_{p\in P}\Profit_p\,x_p \;+\; 300\,s \;-\; 200\,(5\,x_{\text{Widget3}}-s) \\[4pt]
\text{s.t. }&
\sum_{p\in P}\LaborHours_p\,x_p \;\le\; 5000, \\[2pt]
&\sum_{p\in P}\MaterialA_p\,x_p \;\le\; 24000, \\[2pt]
&\sum_{p\in P}\MaterialB_p\,x_p \;\le\; 15000, \\[2pt]
& s \;\le\; 1500, \\[2pt]
& s \;\le\; 5\,x_{\text{Widget3}}, \\[2pt]
& x_p \;\ge\; 0 \;\;\forall p\in P, \qquad s \;\ge\; 0 .
\end{aligned}
\]

*(The objective can be algebraically simplified to  
\(\displaystyle \sum_{p}\Profit_p\,x_p + 500\,s - 1000\,x_{\text{Widget3}}\),  
but the unsimplified form preserves the explicit cost structure.)*

---

### Data Mapping

| Symbol | Source Table | Column(s) | Notes |
|--------|--------------|-----------|-------|
| \(\LaborHours_p\) | `file_1_view_0` | `LaborHours` | per‑unit labor hours for product \(p\) |
| \(\MaterialA_p\) | `file_1_view_0` | `MaterialA` | per‑unit Material A consumption for product \(p\) |
| \(\MaterialB_p\) | `file_1_view_0` | `MaterialB` | per‑unit Material B consumption for product \(p\) |
| \(\Profit_p\) | `file_1_view_0` | `Profit` | base profit per unit of product \(p\) |
| Labor hours limit | `file_0_view_0` | `MonthlyLimit` where `Resource = 'LaborHours'` | 5000 |
| Material A limit | `file_0_view_0` | `MonthlyLimit` where `Resource = 'MaterialA'` | 24000 |
| Material B limit | `file_0_view_0` | `MonthlyLimit` where `Resource = 'MaterialB'` | 15000 |
| CatalystX generation rate | constant | 5 kg per unit of `Widget3` | |
| CatalystX sale price | constant | \$300 per kg | |
| CatalystX disposal cost | constant | \$200 per kg | |
| CatalystX sale cap | constant | 1500 kg | |

All parameter values are taken directly from the complete runtime records of the two CSV files; no additional limits or capacities are imposed beyond those listed.