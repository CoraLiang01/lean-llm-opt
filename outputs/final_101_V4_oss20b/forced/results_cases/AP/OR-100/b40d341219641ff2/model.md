**Mathematical model**

Let  

* \(C\) – set of component types (111 elements)  
* \(W\) – set of workshops (5 elements: Casting, Milling, Finishing, Assembly, QA & Packaging)  

Parameters  

* \(t_{wc}\) – processing time (hours) of component \(c\) in workshop \(w\)  
* \(p_c\) – unit price of component \(c\)  
* \(H_w\) – total available working hours in workshop \(w\)

Decision variables  

* \(q_c \ge 0\) – quantity of component \(c\) produced  

\[
\begin{aligned}
\max\;& \sum_{c\in C} p_c\,q_c \\[4pt]
\text{s.t.}\;& \sum_{c\in C} t_{wc}\,q_c \;\le\; H_w
\qquad \forall\, w\in W \\[4pt]
& q_c \;\ge\; 0 \qquad \forall\, c\in C
\end{aligned}
\]

---

**Data‑mapping rules**

| Parameter | Source table | Row identifier | Column identifier | Mapping expression |
|-----------|--------------|----------------|-------------------|--------------------|
| \(t_{wc}\) | `file_0_view_0` (`processing_time_unit.csv`) | `Unnamed: 0` (workshop name) | `C1`–`C111` (component IDs) | `t_{wc} = value at row where Unnamed: 0 = w, column = Cc` |
| \(p_c\) | `file_1_view_0` (`unit_price.csv`) | `Unnamed: 0` (component ID) | `unit_price` | `p_c = value at row where Unnamed: 0 = c` |
| \(H_w\) | `file_2_view_0` (`total_working_hours.csv`) | `workshop` | `total_hours` | `H_w = value at row where workshop = w` |

All identifiers (`w`, `c`) are taken directly from the corresponding row/column names in the source tables. No numerical values are enumerated in the model; the parameters are linked to the source data via the mapping rules above.