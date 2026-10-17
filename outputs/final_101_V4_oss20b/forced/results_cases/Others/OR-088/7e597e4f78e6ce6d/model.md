## Mathematical Model

Let  

* \(P=\{F1,\dots ,F15\}\) – set of candidate plants  
* \(C=\{C1,\dots ,C15\}\) – set of customers  

Parameters (obtained from the CSV files)

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\text{(opening cost of plant }i)\\
&\text{capacity}_i &&\text{(maximum units plant }i\text{ can supply)}\\
&\text{transport}_{ij} &&\text{(per‑unit transport cost from plant }i\text{ to customer }j)\\
&\text{demand}_j &&\text{(required units for customer }j)
\end{aligned}
\]

Decision variables  

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{(plant }i\text{ is opened)}\\
&x_{ij} &\ge 0 &&\text{(units shipped from plant }i\text{ to customer }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in P}\text{fixed\_cost}_i\,y_i
      +\sum_{i\in P}\sum_{j\in C}\text{transport}_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**

\[
\sum_{i\in P}x_{ij}= \text{demand}_j \qquad \forall j\in C
\]

2. **Plant capacity**

\[
\sum_{j\in C}x_{ij}\;\le\; \text{capacity}_i\,y_i \qquad \forall i\in P
\]

3. **Non‑negativity**

\[
x_{ij}\ge 0 \qquad \forall i\in P,\; j\in C
\]

4. **Binary opening decisions**

\[
y_i\in\{0,1\}\qquad \forall i\in P
\]

---

## Data Mapping

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| `fixed_cost_i` | `file_0_view_0` | `fixed_cost` | `fixed_cost[i]` |
| `capacity_i` | `file_0_view_0` | `capacity` | `capacity[i]` |
| `transport_{ij}` | `file_0_view_0` | `C1`–`C15` | `transport[i,j]` |
| `demand_j` | `file_1_view_0` | `demand` | `demand[j]` |

*Index sets*  
- `P` is the set of plant identifiers (`F1`–`F15`) extracted from the `plant` column of `file_0_view_0`.  
- `C` is the set of customer identifiers (`C1`–`C15`) extracted from the `customer` column of `file_1_view_0`.

All parameters are taken directly from the corresponding CSV rows; no additional scaling or aggregation is performed. The model uses the exact values as provided.