**Mathematical Model**

Let  

* \(S=\{s\}\) be the set of source locations (S1–S10).  
* \(D=\{d\}\) be the set of destination locations (D1–D20).  

Decision variables  

\[
\begin{aligned}
x_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{units shipped from } s \text{ to } d,\\
t_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{number of trucks dispatched on route }(s,d).
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
c_{sd} &\ge 0 &&\text{unit cost from } s \text{ to } d,\\
\text{Supply}_s &\ge 0 &&\text{available units at source } s,\\
\text{Demand}_d &\ge 0 &&\text{required units at destination } d,\\
\text{TruckCap} &= 10 &&\text{maximum units per truck}.
\end{aligned}
\]

---

### Objective  

\[
\min \;\; \sum_{s\in S}\sum_{d\in D} c_{sd}\,x_{sd}
\]

---

### Constraints  

1. **Supply limits**  
   \[
   \sum_{d\in D} x_{sd} \;\le\; \text{Supply}_s , \qquad \forall s\in S
   \]

2. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{sd} \;=\; \text{Demand}_d , \qquad \forall d\in D
   \]

3. **Truck capacity**  
   \[
   x_{sd} \;\le\; \text{TruckCap}\; t_{sd} , \qquad \forall s\in S,\; d\in D
   \]

4. **Non‑negativity & integrality**  
   \[
   x_{sd}\in \mathbb{Z}_{\ge 0},\;\; t_{sd}\in \mathbb{Z}_{\ge 0}, \qquad \forall s,d
   \]

---

## Data Mapping

| Parameter | Source Table | Row ID Column | Column(s) |
|-----------|--------------|---------------|-----------|
| \(c_{sd}\) | `file_0_view_0` | `source_id` | `D1` … `D20` (destination columns) |
| \(\text{Supply}_s\) | `file_2_view_0` | `source_id` | `supply_units` |
| \(\text{Demand}_d\) | `file_1_view_0` | `destination_id` | `demand_units` |
| \(\text{TruckCap}\) | *constant* | – | 10 |

All indices \(s\) and \(d\) are taken directly from the `source_id` and `destination_id` columns of the respective tables. No additional data are introduced beyond what is supplied in the observation.