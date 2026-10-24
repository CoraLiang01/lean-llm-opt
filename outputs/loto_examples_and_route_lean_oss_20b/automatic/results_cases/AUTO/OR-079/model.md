**Mathematical Model**

Let  

- \(F=\{A1,\dots ,A15\}\) (available factory sites)  
- \(D=\{B1,\dots ,B8\}\) (distribution centers)

Parameters  

\[
\begin{aligned}
\text{FixedCost}_f &\quad f\in F &&\text{(fixed cost of building facility }f)\\
\text{Capacity}_f &\quad f\in F &&\text{(maximum units that can be shipped from }f)\\
\text{ShippingCost}_{f,d} &\quad f\in F,\; d\in D &&\text{(unit shipping cost from }f\text{ to }d)\\
\text{Demand}_d &\quad d\in D &&\text{(required units at destination }d)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
y_f &\in \{0,1\}\quad f\in F &&\text{(1 if facility }f\text{ is built)}\\
x_{f,d} &\ge 0 \quad f\in F,\; d\in D &&\text{(units shipped from }f\text{ to }d)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{f\in F}\text{FixedCost}_f\,y_f
      +\sum_{f\in F}\sum_{d\in D}\text{ShippingCost}_{f,d}\,x_{f,d}
\]

**Constraints**

1. **Facility capacity**

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

### Data Mapping

| Parameter | Source Table | Column | Index Set |
|-----------|--------------|--------|-----------|
| \(\text{FixedCost}_f\) | `file_0_view_0` | `FixedCost` | \(f\in F\) |
| \(\text{Capacity}_f\) | `file_0_view_0` | `Capacity` | \(f\in F\) |
| \(\text{ShippingCost}_{f,d}\) | `file_1_view_0` | `B1`–`B8` | \(f\in F,\; d\in D\) |
| \(\text{Demand}_d\) | `file_2_view_0` | `Demand` | \(d\in D\) |

*Index sets \(F\) and \(D\) are defined by the distinct values of `Facility` and `Destination` in the respective tables.*