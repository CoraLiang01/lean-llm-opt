**Mathematical Model**

Let  

- \(I=\{F1,\dots ,F15\}\) (plant indices)  
- \(J=\{C1,\dots ,C15\}\) (customer indices)

Parameters  

\[
\begin{aligned}
&\text{fixed\_cost}_{i}\quad &&\text{(opening cost of plant }i)\\
&\text{capacity}_{i}\quad &&\text{(maximum units plant }i\text{ can supply)}\\
&\text{transport\_cost}_{ij}\quad &&\text{(per‑unit shipping cost from plant }i\text{ to customer }j)\\
&\text{demand}_{j}\quad &&\text{(required units for customer }j)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i \in \{0,1\}\quad &&\text{(1 if plant }i\text{ is opened)}\\
&x_{ij}\in \mathbb{Z}_{\ge 0}\quad &&\text{(units shipped from plant }i\text{ to customer }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{fixed\_cost}_{i}\,y_i
      +\sum_{i\in I}\sum_{j\in J}\text{transport\_cost}_{ij}\,x_{ij}
\]

**Constraints**

1. **Capacity of opened plants**

\[
\sum_{j\in J}x_{ij}\;\le\;\text{capacity}_{i}\,y_i
\qquad \forall i\in I
\]

2. **Demand satisfaction**

\[
\sum_{i\in I}x_{ij}\;\ge\;\text{demand}_{j}
\qquad \forall j\in J
\]

3. **Variable domains**

\[
x_{ij}\in \mathbb{Z}_{\ge 0}\;\;\forall i\in I,\;j\in J,
\qquad
y_i\in\{0,1\}\;\;\forall i\in I
\]

---

**Data Mapping**

| Parameter / Variable | Source Table | Column(s) | Notes |
|----------------------|--------------|-----------|-------|
| `fixed_cost_i` | `file_0_view_0` | `fixed_cost` | Plant opening cost |
| `capacity_i` | `file_0_view_0` | `capacity` | Plant capacity |
| `transport_cost_ij` | `file_0_view_0` | `C1`–`C15` | Per‑unit shipping cost |
| `demand_j` | `file_1_view_0` | `demand` | Customer demand |
| `y_i` | — | — | Binary decision variable |
| `x_ij` | — | — | Integer shipment variable |

All indices \(i\) and \(j\) are taken from the rows of the respective tables (`plant` column in `file_0_view_0` and `customer` column in `file_1_view_0`). No values are hard‑coded; the model references the data via the table and column names above.