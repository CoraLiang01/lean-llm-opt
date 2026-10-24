**Mathematical model**

Let  

* \(C=\{C1,\dots ,C111\}\) – set of component types.  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – set of workshops.  

Parameters  

* \(p_c\) – unit price of component \(c\) (dollars).  
* \(t_{w,c}\) – processing time (hours) required for component \(c\) in workshop \(w\).  
* \(H_w\) – total available working hours in workshop \(w\).  

Decision variables  

* \(x_c\in \mathbb{Z}_{\ge 0}\) – number of units of component \(c\) produced.  

\[
\begin{aligned}
\max\;&\sum_{c\in C} p_c\,x_c \\[4pt]
\text{s.t. } &\sum_{c\in C} t_{w,c}\,x_c \;\le\; H_w , &&\forall\, w\in\{\text{Casting},\text{Milling},\text{Assembly},\text{QA \& Packaging}\} \\[4pt]
&x_c \in \mathbb{Z}_{\ge 0}, &&\forall\, c\in C .
\end{aligned}
\]

*No capacity constraint is imposed for the “Finishing” workshop because the data set does not provide a total‑hours value for it.*

---

**Data‑mapping rules**

| Parameter | Source table | Key column(s) | Value column |
|-----------|--------------|---------------|--------------|
| \(p_c\) | `file_1_view_0` | `Unnamed: 0` (component id) | `unit_price` |
| \(t_{w,c}\) | `file_0_view_0` | Row: `workshop` (workshop id) <br> Column: `Unnamed: 0` (component id) | Cell value (processing time) |
| \(H_w\) | `file_2_view_0` | `workshop` (workshop id) | `total_hours` |

All indices and coefficients are taken directly from the runtime records supplied in the observation; no additional data are invented.