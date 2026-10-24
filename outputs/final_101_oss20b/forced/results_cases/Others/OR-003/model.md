**Mathematical Model**

Let  

\[
S=\{S1,\dots ,S10\},\qquad C=\{C1,\dots ,C10\}
\]

Decision variables  

\[
x_{s,c}\;\ge 0\qquad \forall s\in S,\;c\in C
\]

Parameters  

\[
\begin{aligned}
\text{cap}_s      &\;=\;\text{supply_capacity of supplier }s\\
\text{dem}_c      &\;=\;\text{demand of customer }c\\
\text{cost}_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c
\end{aligned}
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}_{s,c}\,x_{s,c}
\]

Supply‑capacity constraints  

\[
\sum_{c\in C}x_{s,c}\;\le\;\text{cap}_s \qquad \forall s\in S
\]

Demand‑fulfilment constraints  

\[
\sum_{s\in S}x_{s,c}\;=\;\text{dem}_c \qquad \forall c\in C
\]

All variables are continuous and non‑negative.

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| `cap_s`   | `file_1_view_0` | `supply_capacity` |
| `dem_c`   | `file_0_view_0` | `demand` |
| `cost_{s,c}` | `file_2_view_0` | `C1`–`C10` (row identified by `Unnamed: 0` = supplier) |

Index sets `S` and `C` are taken from the distinct values of the supplier and customer identifiers present in the respective tables. No additional filtering or transformation is applied.