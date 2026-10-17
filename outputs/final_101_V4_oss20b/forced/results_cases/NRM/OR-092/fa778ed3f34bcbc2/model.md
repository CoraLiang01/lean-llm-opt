**Mathematical Model**

Let  

* \(S=\{S_1,\dots ,S_{10}\}\) – source locations  
* \(D=\{D_1,\dots ,D_{20}\}\) – destination locations  

Parameters  

\[
\begin{aligned}
\text{Supply}_s &\quad\text{(units available at source }s)\\
\text{Demand}_d &\quad\text{(units required at destination }d)\\
c_{sd} &\quad\text{(unit transportation cost from }s\text{ to }d)\\
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{units shipped from }s\text{ to }d\\
y_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{number of trucks dispatched from }s\text{ to }d
\end{aligned}
\]

**Objective**

\[
\min \;\; \sum_{s\in S}\sum_{d\in D} c_{sd}\,x_{sd}
\]

**Constraints**

1. **Truck capacity (partial loading allowed)**  

\[
x_{sd}\;\le\;10\,y_{sd}\qquad \forall s\in S,\;d\in D
\]

2. **Supply limits**  

\[
\sum_{d\in D} x_{sd}\;\le\;\text{Supply}_s \qquad \forall s\in S
\]

3. **Demand satisfaction**  

\[
\sum_{s\in S} x_{sd}\;\ge\;\text{Demand}_d \qquad \forall d\in D
\]

4. **Variable domains**  

\[
x_{sd}\in\mathbb{Z}_{\ge 0},\;\; y_{sd}\in\mathbb{Z}_{\ge 0}\qquad \forall s,d
\]

---

### Data Mapping

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\text{Supply}_s\) | `file_2_view_0` | `supply_units` |
| \(\text{Demand}_d\) | `file_1_view_0` | `demand_units` |
| \(c_{sd}\) | `file_0_view_0` | `D1`–`D20` (columns indexed by destination) |

All indices \(s\) and \(d\) are taken from the `source_id` and `destination_id` columns of the respective tables. No additional data are introduced beyond what is supplied in the observation.