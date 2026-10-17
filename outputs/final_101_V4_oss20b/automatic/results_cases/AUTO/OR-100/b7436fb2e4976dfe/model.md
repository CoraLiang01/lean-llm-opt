**Mathematical Model**

Let  

* \(C=\{C1,\dots ,C111\}\) – set of component types.  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – set of workshops.

Parameters  

\[
\begin{aligned}
p_c &\;=\;\text{unit price of component }c, && c\in C,\\
t_{wc}&\;=\;\text{processing time (hours) of component }c\text{ in workshop }w, && w\in W,\;c\in C,\\
H_w &\;=\;\text{total available working hours in workshop }w, && w\in W\text{ for which data exist}.
\end{aligned}
\]

Decision variables  

\[
x_c \;\in\; \mathbb{Z}_{\ge 0}\qquad (c\in C)
\]

---

**Objective**

\[
\max \;\sum_{c\in C} p_c\,x_c
\]

---

**Constraints**

1. **Workshop capacity** – for every workshop \(w\) that has a total‑hour value:

\[
\sum_{c\in C} t_{wc}\,x_c \;\le\; H_w , \qquad w\in W_{\text{hours}}
\]

where \(W_{\text{hours}}\subseteq W\) is the subset of workshops with a defined \(H_w\) (Finishing is omitted because no total‑hour entry is provided).

2. **Non‑negativity and integrality**

\[
x_c \;\ge\; 0,\qquad x_c \in \mathbb{Z}, \qquad \forall\,c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Key Column(s) | Value Column |
|-----------|--------------|---------------|--------------|
| \(p_c\) | `file_1_view_0` | `Unnamed: 0` (component ID) | `unit_price` |
| \(t_{wc}\) | `file_0_view_0` | `Unnamed: 0` (workshop name) as row key, `C1`–`C111` as column keys | corresponding cell value |
| \(H_w\) | `file_2_view_0` | `workshop` | `total_hours` |

*Index sets*  
- \(C\) is derived from the 111 component identifiers present in `file_1_view_0`.  
- \(W\) is derived from the workshop names in the row identifiers of `file_0_view_0`.  
- \(W_{\text{hours}}\) is the subset of \(W\) for which `file_2_view_0` contains a `total_hours` entry.

All coefficients are taken directly from the corresponding CSV cells; no additional data are invented. The model respects the integer domain for production quantities and the capacity limits of the workshops that have available hours.