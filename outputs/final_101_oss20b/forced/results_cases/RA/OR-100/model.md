**Mathematical model**

Let  

* \(C=\{C1,\dots ,C111\}\) – set of component types.  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – set of workshops.  

Parameters  

* \(p_c\) – unit price of component \(c\).  
* \(t_{wc}\) – processing time (hours) of component \(c\) in workshop \(w\).  
* \(H_w\) – total available working hours in workshop \(w\) (only for workshops with data).  

Decision variables  

* \(x_c\in\mathbb{Z}_{\ge 0}\) – number of units of component \(c\) produced.  

\[
\begin{aligned}
\max\;&\sum_{c\in C} p_c\,x_c \\[4pt]
\text{s.t.}\;&\sum_{c\in C} t_{wc}\,x_c \;\le\; H_w , \qquad
w\in\{\text{Casting},\text{Milling},\text{Assembly},\text{QA \& Packaging}\} \\[4pt]
&x_c\in\mathbb{Z}_{\ge 0}, \qquad c\in C .
\end{aligned}
\]

---

**Data mapping (CSVQA_DATA)**  

| Parameter | Source table | Column(s) | Mapping rule |
|-----------|--------------|-----------|--------------|
| \(t_{wc}\) | `file_0_view_0` | matrix entries (row = workshop, column = component) | `t[w,c] = CSVQA_DATA.file_0_view_0[w][c]` |
| \(p_c\) | `file_1_view_0` | `unit_price` | `p[c] = CSVQA_DATA.file_1_view_0[c]['unit_price']` |
| \(H_w\) | `file_2_view_0` | `total_hours` | `H[w] = CSVQA_DATA.file_2_view_0[w]['total_hours']` |

*Only workshops with a `total_hours` entry are constrained.*