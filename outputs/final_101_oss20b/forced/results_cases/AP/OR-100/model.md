**Mathematical Model**

Let  

* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\)  
* \(C=\{C1,C2,\dots ,C111\}\)

Parameters  

* \(t_{w,c}\) – unit processing time of component \(c\) in workshop \(w\)  
* \(p_c\) – unit price of component \(c\)  
* \(H_w\) – total available working hours in workshop \(w\)

Decision variables  

* \(x_c \in \mathbb{Z}_{+}\) – number of units of component \(c\) produced

---

\[
\begin{aligned}
\max\;& \sum_{c\in C} p_c\,x_c \\[4pt]
\text{s.t.}\;& \sum_{c\in C} t_{w,c}\,x_c \;\le\; H_w
      &&\forall\, w\in W \\[4pt]
& x_c \;\in\; \mathbb{Z}_{+} &&\forall\, c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Row Identifier | Column Identifier | Notes |
|-----------|---------------|----------------|--------------------|-------|
| \(t_{w,c}\) | `file_0_view_0` | `Unnamed: 0` (workshop name) | `C1`–`C111` (component) | 5 × 111 matrix |
| \(p_c\) | `file_1_view_0` | `Unnamed: 0` (component) | `unit_price` | 111 × 1 vector |
| \(H_w\) | `file_2_view_0` | `workshop` | `total_hours` | 5 × 1 vector |

All identifiers are taken exactly as they appear in the source tables; no additional enumeration or transformation is performed. The model uses the full set of workshops and components returned by the observation.