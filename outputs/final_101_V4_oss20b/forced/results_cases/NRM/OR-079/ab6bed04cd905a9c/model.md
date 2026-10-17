**Symbolic Model**

Let  

- \(F=\{A1,\dots ,A15\}\) : set of potential factory sites  
- \(D=\{B1,\dots ,B8\}\) : set of distribution centers  

Parameters  

\[
\begin{aligned}
&\text{FixedCost}_f &&\forall f\in F \\
&\text{Capacity}_f   &&\forall f\in F \\
&\text{ShippingCost}_{f,d} &&\forall f\in F,\; d\in D \\
&\text{Demand}_d &&\forall d\in D
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_f \in \{0,1\} &&\text{(factory \(f\) is built)}\\
&x_{f,d}\;\ge 0 &&\text{(units shipped from \(f\) to \(d\))}
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{f\in F}\text{FixedCost}_f\,y_f
      +\sum_{f\in F}\sum_{d\in D}\text{ShippingCost}_{f,d}\,x_{f,d}
\]

**Constraints**

1. **Factory capacity**

\[
\sum_{d\in D}x_{f,d}\;\le\;\text{Capacity}_f\,y_f
\qquad \forall f\in F
\]

2. **Demand satisfaction**

\[
\sum_{f\in F}x_{f,d}\;=\;\text{Demand}_d
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

| Parameter | Source Table | Source Columns |
|-----------|--------------|----------------|
| `FixedCost_f` | `file_0_view_0` | `Facility`, `FixedCost` |
| `Capacity_f` | `file_0_view_0` | `Facility`, `Capacity` |
| `ShippingCost_{f,d}` | `file_1_view_0` | `Origin` (facility), `B1`–`B8` (destinations) |
| `Demand_d` | `file_2_view_0` | `Destination`, `Demand` |

All indices \(f\) and \(d\) are taken from the corresponding tables’ identifier columns (`Facility` and `Destination`). No additional data are introduced beyond what is supplied in the three CSV files.