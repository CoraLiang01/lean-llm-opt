**Mathematical Model**

Let  

\[
I=\{\text{Widget1},\dots ,\text{Widget141}\}
\]

Decision variables  

\[
x_i\ge 0 \qquad(i\in I) \quad\text{(units of widget }i\text{ produced)}
\]
\[
s\ge 0 \qquad\text{(kg of CatalystX sold)}
\]

Parameters  

\[
\begin{aligned}
&\text{LH}_i   &&\text{Labor hours per unit of widget }i\\
&\text{MA}_i   &&\text{Material A (kg) per unit of widget }i\\
&\text{MB}_i   &&\text{Material B (kg) per unit of widget }i\\
&\text{P}_i    &&\text{Base profit ($) per unit of widget }i\\
&\text{C}_{\text{prod}}   &&5\;\text{kg of CatalystX per unit of Widget3}\\
&\text{C}_{\text{sell}}   &&300\;\text{\$/kg (sale price)}\\
&\text{C}_{\text{dispose}}&&200\;\text{\$/kg (disposal cost)}\\
&\text{LH}^{\max}   &&5000\;\text{labor hours per month}\\
&\text{MA}^{\max}   &&24000\;\text{kg of Material A per month}\\
&\text{MB}^{\max}   &&15000\;\text{kg of Material B per month}\\
&\text{S}^{\max}    &&1500\;\text{kg of CatalystX that can be sold}
\end{aligned}
\]

Objective  

\[
\max \;\sum_{i\in I}\text{P}_i\,x_i \;-\;1000\,x_{\text{Widget3}}\;+\;500\,s
\]

(Profit from widgets, minus disposal cost of unsold CatalystX, plus revenue from sold CatalystX.)

Constraints  

\[
\begin{aligned}
&\sum_{i\in I}\text{LH}_i\,x_i \;\le\; \text{LH}^{\max} \qquad\text{(labor)}\\[4pt]
&\sum_{i\in I}\text{MA}_i\,x_i \;\le\; \text{MA}^{\max} \qquad\text{(Material A)}\\[4pt]
&\sum_{i\in I}\text{MB}_i\,x_i \;\le\; \text{MB}^{\max} \qquad\text{(Material B)}\\[4pt]
& s \;\le\; \text{S}^{\max} \qquad\text{(market cap on CatalystX)}\\[4pt]
& s \;\le\; 5\,x_{\text{Widget3}} \qquad\text{(cannot sell more than produced)}\\[4pt]
& x_i \ge 0,\; s \ge 0 \qquad\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(I\) | file_1_view_0 | Product | All 141 product identifiers |
| \(\text{LH}_i\) | file_1_view_0 | LaborHours | Float values |
| \(\text{MA}_i\) | file_1_view_0 | MaterialA | Integer values |
| \(\text{MB}_i\) | file_1_view_0 | MaterialB | Integer values |
| \(\text{P}_i\) | file_1_view_0 | Profit | Integer values |
| \(\text{LH}^{\max}\) | file_0_view_0 | MonthlyLimit | where Resource = LaborHours |
| \(\text{MA}^{\max}\) | file_0_view_0 | MonthlyLimit | where Resource = MaterialA |
| \(\text{MB}^{\max}\) | file_0_view_0 | MonthlyLimit | where Resource = MaterialB |
| \(\text{S}^{\max}\) | constant | – | 1500 kg |
| \(\text{C}_{\text{prod}}\) | constant | – | 5 kg per Widget3 unit |
| \(\text{C}_{\text{sell}}\) | constant | – | 300 $/kg |
| \(\text{C}_{\text{dispose}}\) | constant | – | 200 $/kg |

All other constants (e.g., 1000, 500 in the objective) arise from algebraic manipulation of the sale and disposal terms and are not directly mapped to source data. The model uses continuous, non‑negative decision variables as specified.