**Mathematical Model**

Let  

* \(S=\{S1,\dots ,S10\}\) – source locations  
* \(D=\{D1,\dots ,D20\}\) – destination locations  

Decision variables  

\[
\begin{aligned}
u_{s,d} &\ge 0 &&\text{(units of cargo shipped from }s\text{ to }d)\\
t_{s,d} &\in \mathbb{Z}_{\ge 0} &&\text{(number of trucks dispatched from }s\text{ to }d)
\end{aligned}
\]

Parameters  

\[
\begin{aligned}
c_{s,d} &\ge 0 &&\text{unit cost on route }(s,d)\\
\text{Supply}_s &\ge 0 &&\text{available units at source }s\\
\text{Demand}_d &\ge 0 &&\text{required units at destination }d
\end{aligned}
\]

---

**Objective**

\[
\min \;\; \sum_{s\in S}\sum_{d\in D} c_{s,d}\,u_{s,d}
\]

---

**Constraints**

1. **Supply limits**  
   \[
   \sum_{d\in D} u_{s,d}\;\le\;\text{Supply}_s \qquad \forall s\in S
   \]

2. **Demand satisfaction**  
   \[
   \sum_{s\in S} u_{s,d}\;=\;\text{Demand}_d \qquad \forall d\in D
   \]

3. **Truck capacity (partial loading allowed)**  
   \[
   u_{s,d}\;\le\;10\,t_{s,d} \qquad \forall s\in S,\;d\in D
   \]

4. **Variable domains**  
   \[
   u_{s,d}\;\ge\;0,\qquad t_{s,d}\;\in\mathbb{Z}_{\ge 0}\qquad \forall s,d
   \]

---

## Data Mapping

| Parameter | Source Table | Column(s) | Notes |
|-----------|--------------|-----------|-------|
| \(c_{s,d}\) | `file_0_view_0` (expanded_cost_matrix.csv) | `D1`–`D20` | Cost per unit for each source‑destination pair |
| \(\text{Supply}_s\) | `file_2_view_0` (expanded_sources.csv) | `supply_units` | Units available at each source |
| \(\text{Demand}_d\) | `file_1_view_0` (expanded_destinations.csv) | `demand_units` | Units required at each destination |

All indices \(s\) and \(d\) are taken from the complete sets of rows in the respective tables; no additional filtering is applied. The model uses the full 10×20 cost matrix, the 10 supply values, and the 20 demand values as provided.