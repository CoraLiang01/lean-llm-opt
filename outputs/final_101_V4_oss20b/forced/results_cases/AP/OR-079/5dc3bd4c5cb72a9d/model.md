**Mathematical Model**

Let  

\[
F=\{A1,\dots ,A15\}, \qquad D=\{B1,\dots ,B8\}
\]

Parameters  

\[
\begin{aligned}
&\text{FixedCost}_f &&\forall f\in F \\
&\text{Capacity}_f   &&\forall f\in F \\
&\text{Demand}_d     &&\forall d\in D \\
&\text{ShippingCost}_{f,d} &&\forall f\in F,\; d\in D
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_f &\in\{0,1\} &&\forall f\in F \quad (\text{facility built})\\
&x_{f,d} &\ge 0 &&\forall f\in F,\; d\in D \quad (\text{units shipped})
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{f\in F}\text{FixedCost}_f\,y_f
      +\sum_{f\in F}\sum_{d\in D}\text{ShippingCost}_{f,d}\,x_{f,d}
\]

**Constraints**

1. **Capacity**

\[
\sum_{d\in D}x_{f,d}\;\le\;\text{Capacity}_f\,y_f
\qquad \forall f\in F
\]

2. **Demand satisfaction**

\[
\sum_{f\in F}x_{f,d}\;\ge\;\text{Demand}_d
\qquad \forall d\in D
\]

3. **Non‑negativity**

\[
x_{f,d}\;\ge\;0 \qquad \forall f\in F,\; d\in D
\]

4. **Binary facility decision**

\[
y_f\in\{0,1\}\qquad \forall f\in F
\]

---

**Data Mapping**

| Parameter | Source Table | Column / Matrix |
|-----------|--------------|-----------------|
| FixedCost_f | `file_0_view_0` | `FixedCost` |
| Capacity_f | `file_0_view_0` | `Capacity` |
| Demand_d | `file_2_view_0` | `Demand` |
| ShippingCost_{f,d} | `file_1_view_0` | matrix with rows `Origin` (facilities) and columns `B1`–`B8` |

All index sets \(F\) and \(D\) are taken directly from the row and column identifiers of the respective tables. No numerical values are enumerated in the model.