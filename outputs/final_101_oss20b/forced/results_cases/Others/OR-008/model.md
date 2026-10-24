## Mathematical Model

Let  

* \(S=\{\text{Supplier1},\dots ,\text{Supplier5}\}\)  
* \(C=\{\text{Customer1},\dots ,\text{Customer6}\}\)

Decision variables  

\[
x_{s,c}\;\;(\text{units of produce shipped from }s\text{ to }c),\qquad
x_{s,c}\ge 0,\; s\in S,\;c\in C
\]

Parameters  

\[
\begin{aligned}
d_c &\;=\;\text{daily demand of customer }c,\\
u_s &\;=\;\text{available supply capacity of supplier }s,\\
c_{s,c} &\;=\;\text{transportation cost per unit from }s\text{ to }c .
\end{aligned}
\]

### Objective  

\[
\min \;\; Z=\sum_{s\in S}\sum_{c\in C} c_{s,c}\,x_{s,c}
\]

### Constraints  

1. **Demand satisfaction** – each customer’s demand must be met exactly  

\[
\sum_{s\in S} x_{s,c}=d_c ,\qquad \forall\,c\in C
\]

2. **Supply capacity** – a supplier cannot ship more than its capacity  

\[
\sum_{c\in C} x_{s,c}\le u_s ,\qquad \forall\,s\in S
\]

3. **Non‑negativity**  

\[
x_{s,c}\ge 0 ,\qquad \forall\,s\in S,\;c\in C
\]

---

## Data Mapping

| Parameter | Source | Table ID | Row/Column Axis | Column |
|-----------|--------|----------|-----------------|--------|
| \(d_c\) | Customer demand | `file_0_view_0` | `Customers` | `demand` |
| \(u_s\) | Supplier capacity | `file_1_view_0` | `Suppliers` | `supply_capacity` |
| \(c_{s,c}\) | Transportation cost | `file_2_view_0` | Row: `Suppliers` (`Unnamed: 0`) | Column: `Customers` (`Customer1`–`Customer6`) |

| Index set | Source | Table ID | Column |
|-----------|--------|----------|--------|
| \(S\) | Supplier list | `file_1_view_0` | `Suppliers` |
| \(C\) | Customer list | `file_0_view_0` | `Customers` |

All values are taken directly from the supplied CSV files; no additional numerical values are introduced. The model is a linear program with continuous shipment variables and linear constraints as specified.