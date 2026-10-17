**Mathematical Model**

Let  

\[
I=\{\,i\mid i \text{ is a ProductName in } \text{products.csv}\,\}
\]

Parameters  

\[
\begin{aligned}
&\text{Value}_i &&\text{benefit per unit of area }i,\\
&\text{Weight}_i &&\text{capacity consumption per unit of area }i,\\
&\text{Capacity} &&\text{overall development capacity (from capacity.csv)}.
\end{aligned}
\]

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\quad \forall i\in I
\]

Objective  

\[
\max \;\sum_{i\in I} \text{Value}_i\,x_i
\]

Capacity constraint  

\[
\sum_{i\in I} \text{Weight}_i\,x_i \;\le\; \text{Capacity}
\]

Non‑negativity (implicit in integer domain)  

\[
x_i \ge 0 \quad \forall i\in I
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| `Capacity` | `file_0_view_0` | `Capacity` |
| `Value_i` | `file_1_view_0` | `Value` |
| `Weight_i` | `file_1_view_0` | `Weight` |
| `i` (index) | `file_1_view_0` | `ProductName` |

All parameter values are taken directly from the corresponding columns of the specified tables; no additional data or transformations are introduced.