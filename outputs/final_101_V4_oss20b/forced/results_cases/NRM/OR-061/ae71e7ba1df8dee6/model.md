**Mathematical Model**

Let  

- \(I=\{S_1,S_2,S_3,S_4,S_5\}\) : set of suppliers  
- \(J=\{C_1,C_2,C_3,C_4,C_5\}\) : set of customers  

**Parameters**

\[
\begin{aligned}
&\text{fixed\_cost}_i &&\ge 0 &&\text{(fixed cost of opening supplier }i)\\
&\text{demand}_j &&\ge 0 &&\text{(demand of customer }j)\\
&\text{transport\_cost}_{ij} &&\ge 0 &&\text{(unit transport cost from }i\text{ to }j)\\
&M &&\ge 0 &&\text{big‑\(M\) (e.g. }M=\sum_{j\in J}\text{demand}_j\text{)}
\end{aligned}
\]

**Decision Variables**

\[
\begin{aligned}
&y_i &\in\{0,1\} &&\text{(1 if supplier }i\text{ is opened)}\\
&x_{ij} &\ge 0 &&\text{(quantity shipped from }i\text{ to }j)
\end{aligned}
\]

**Objective**

\[
\min \;\sum_{i\in I}\text{fixed\_cost}_i\,y_i
      \;+\;\sum_{i\in I}\sum_{j\in J}\text{transport\_cost}_{ij}\,x_{ij}
\]

**Constraints**

1. **Demand satisfaction**  
   \[
   \sum_{i\in I} x_{ij}\;\ge\;\text{demand}_j
   \qquad \forall j\in J
   \]

2. **Supplier activation (big‑\(M\) link)**  
   \[
   x_{ij}\;\le\; M\,y_i
   \qquad \forall i\in I,\; \forall j\in J
   \]

---

### Data Mapping

| Parameter | Source Table | Key Column | Value Column |
|-----------|--------------|------------|--------------|
| `fixed_cost_i` | `file_1_view_0` | `Unnamed: 0` (supplier id) | `fixed_costs` |
| `demand_j` | `file_0_view_0` | `customer` | `demand` |
| `transport_cost_{ij}` | `file_2_view_0` | `Unnamed: 0` (supplier id) | `C1`–`C5` (customer columns) |
| `M` | *derived* | – | \(\sum_{j\in J}\text{demand}_j\) |

All numerical values are taken directly from the corresponding CSV files; no additional data are introduced.