**Mathematical Model**

Let  

* \(S=\{s\}\) be the set of source locations (S1–S10).  
* \(D=\{d\}\) be the set of destination locations (D1–D20).  

Parameters  

\[
\begin{aligned}
\text{Supply}_s &\;=\; \text{supply_units from file\_2\_view\_0 for source }s,\\
\text{Demand}_d &\;=\; \text{demand_units from file\_1\_view\_0 for destination }d,\\
c_{sd} &\;=\; \text{unit cost from file\_0\_view\_0 for source }s\text{ to destination }d,\\
\text{TruckCap} &= 10 \quad(\text{units per truck}).
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
x_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{units shipped from }s\text{ to }d,\\
t_{sd} &\in \mathbb{Z}_{\ge 0} &&\text{number of trucks dispatched on route }(s,d).
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{s\in S}\sum_{d\in D} c_{sd}\,x_{sd}
\]

**Constraints**

1. **Supply limits**  
   \[
   \sum_{d\in D} x_{sd} \;\le\; \text{Supply}_s ,\qquad \forall s\in S
   \]

2. **Demand satisfaction**  
   \[
   \sum_{s\in S} x_{sd} \;=\; \text{Demand}_d ,\qquad \forall d\in D
   \]

3. **Truck capacity**  
   \[
   x_{sd} \;\le\; \text{TruckCap}\; t_{sd}, \qquad \forall s\in S,\; d\in D
   \]

4. **Non‑negativity & integrality**  
   \[
   x_{sd}\in\mathbb{Z}_{\ge0},\; t_{sd}\in\mathbb{Z}_{\ge0}\quad \forall s,d
   \]

---

### Data Mapping

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| `Supply_s` | `file_2_view_0` | `supply_units` (indexed by `source_id`) |
| `Demand_d` | `file_1_view_0` | `demand_units` (indexed by `destination_id`) |
| `c_{sd}`   | `file_0_view_0` | `D1`–`D20` columns (indexed by `source_id` and destination column) |
| `TruckCap` | — | constant value 10 |

All indices (`S`, `D`) are taken directly from the `source_id` and `destination_id` columns of the respective tables. No additional data are introduced beyond what is supplied.