**Mathematical Model**

Let  

- \(S=\{S_1,\dots ,S_{10}\}\) : source locations  
- \(D=\{D_1,\dots ,D_{20}\}\) : destination locations  

Parameters  

\[
\begin{aligned}
&\text{Supply}_s &&\forall s\in S &&\text{(units available at source }s)\\
&\text{Demand}_d &&\forall d\in D &&\text{(units required at destination }d)\\
&c_{sd} &&\forall s\in S,\;d\in D &&\text{(unit cost from }s\text{ to }d)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&x_{sd} &&\in \mathbb{Z}_{+} &&\text{(number of trucks dispatched from }s\text{ to }d)\\
&u_{sd} &&\in \mathbb{Z}_{+} &&\text{(units shipped from }s\text{ to }d)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{d\in D} c_{sd}\,u_{sd}
\]

**Constraints**

1. **Truck capacity (partial loading allowed)**  

\[
u_{sd}\;\le\;10\,x_{sd}\qquad \forall s\in S,\;d\in D
\]

2. **Supply limits**  

\[
\sum_{d\in D} u_{sd}\;\le\;\text{Supply}_s \qquad \forall s\in S
\]

3. **Demand satisfaction**  

\[
\sum_{s\in S} u_{sd}\;=\;\text{Demand}_d \qquad \forall d\in D
\]

4. **Variable domains**  

\[
x_{sd}\in\mathbb{Z}_{+},\;u_{sd}\in\mathbb{Z}_{+}\qquad \forall s\in S,\;d\in D
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(\text{Supply}_s\) | `file_2_view_0` | `supply_units` (indexed by `source_id`) |
| \(\text{Demand}_d\) | `file_1_view_0` | `demand_units` (indexed by `destination_id`) |
| \(c_{sd}\) | `file_0_view_0` | `D1`–`D20` columns (indexed by `source_id`) |

All indices \(s\) and \(d\) are taken from the respective `source_id` and `destination_id` values present in the tables. No additional data are required.