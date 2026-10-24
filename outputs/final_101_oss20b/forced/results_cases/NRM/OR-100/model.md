**Mathematical Model**

Let  

* \(C=\{C1,\dots ,C111\}\) – set of component types  
* \(W=\{\text{Casting},\text{Milling},\text{Finishing},\text{Assembly},\text{QA \& Packaging}\}\) – set of workshops  

Parameters  

\[
\begin{aligned}
p_{w,c} &\;=\;\text{processing time (hours) of one unit of component }c\text{ in workshop }w,\\
price_{c} &\;=\;\text{unit price of component }c,\\
H_{w} &\;=\;\text{total available working hours in workshop }w.
\end{aligned}
\]

Decision variables  

\[
x_{c}\;\in\;\mathbb{Z}_{+}\qquad \forall\,c\in C
\]

---

**Objective**

\[
\max \;\sum_{c\in C} price_{c}\;x_{c}
\]

---

**Constraints**

1. **Workshop capacity**

\[
\sum_{c\in C} p_{w,c}\;x_{c}\;\le\; H_{w}\qquad \forall\,w\in W
\]

2. **Non‑negativity & integrality**

\[
x_{c}\;\ge\;0,\qquad x_{c}\in\mathbb{Z}\qquad \forall\,c\in C
\]

---

**Data Mapping**

| Parameter | Source Table ID | Source Column(s) | Notes |
|-----------|-----------------|------------------|-------|
| \(p_{w,c}\) | `file_0_view_0` | Row: `workshop` (values: Casting, Milling, Finishing, Assembly, QA & Packaging); Column: `C1`–`C111` | Each cell gives processing time for component \(c\) in workshop \(w\). |
| \(price_{c}\) | `file_1_view_0` | Column: `unit_price` | Row identified by `Unnamed: 0` = component code (`C1`–`C111`). |
| \(H_{w}\) | `file_2_view_0` | Column: `total_hours` | Row identified by `workshop`. (Finishing row is missing in the provided data; if present, use its value.) |

All other data (e.g., component identifiers) are taken directly from the corresponding rows/columns of the source tables. No additional data or assumptions are introduced beyond what is supplied.