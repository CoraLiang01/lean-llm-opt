**Mathematical Model**

\[
\begin{aligned}
\max\;&\sum_{i\in I} \text{Profit}_i\,x_i \;+\; 300\,s \;-\; 200\,d \\[4pt]
\text{s.t. }&
\sum_{i\in I} \text{LaborHours}_i\,x_i \;\le\; L_{\text{LaborHours}} \\[2pt]
&\sum_{i\in I} \text{MaterialA}_i\,x_i \;\le\; L_{\text{MaterialA}} \\[2pt]
&\sum_{i\in I} \text{MaterialB}_i\,x_i \;\le\; L_{\text{MaterialB}} \\[4pt]
& s + d \;=\; 5\,x_{\text{Widget3}} \\[2pt]
& s \;\le\; 1500 \\[2pt]
& x_i \;\in\;\mathbb{Z}_{\ge 0}\quad \forall i\in I \\[2pt]
& s,d \;\ge\; 0
\end{aligned}
\]

*Indices*  
\(I\) – set of all products \(\{ \text{Widget1},\dots,\text{Widget141}\}\).

*Parameters*  
\(\text{LaborHours}_i,\;\text{MaterialA}_i,\;\text{MaterialB}_i,\;\text{Profit}_i\) – per‑unit data for product \(i\).  
\(L_{\text{LaborHours}},\;L_{\text{MaterialA}},\;L_{\text{MaterialB}}\) – monthly resource limits.

*Decision variables*  
\(x_i\) – number of units of product \(i\) produced (integer).  
\(s\) – kilograms of CatalystX sold (continuous).  
\(d\) – kilograms of CatalystX disposed (continuous).

---

**Data Mapping**

| Parameter / Variable | Table ID | Column(s) | Notes |
|----------------------|----------|-----------|-------|
| \(\text{LaborHours}_i\) | `file_1_view_0` | `LaborHours` | per product |
| \(\text{MaterialA}_i\) | `file_1_view_0` | `MaterialA` | per product |
| \(\text{MaterialB}_i\) | `file_1_view_0` | `MaterialB` | per product |
| \(\text{Profit}_i\) | `file_1_view_0` | `Profit` | per product |
| \(L_{\text{LaborHours}}\) | `file_0_view_0` | `MonthlyLimit` | where `Resource = 'LaborHours'` |
| \(L_{\text{MaterialA}}\) | `file_0_view_0` | `MonthlyLimit` | where `Resource = 'MaterialA'` |
| \(L_{\text{MaterialB}}\) | `file_0_view_0` | `MonthlyLimit` | where `Resource = 'MaterialB'` |

All other data (e.g., the 5 kg CatalystX per unit of Widget3, sale price 300 $/kg, disposal cost 200 $/kg, and sale cap 1500 kg) are constants defined in the model formulation.