**Mathematical Model**

Let  

* \(I=\{P1,\dots ,P100\}\) (Products)  
* \(R=\{R1,R2,R3\}\) (Resources)

Parameters (obtained from the CSV data)

| Symbol | Definition | Source |
|--------|------------|--------|
| \(p_i\) | Profit per unit of product \(i\) | `file_0_view_0.profit_per_unit` |
| \(a_{ri}\) | Consumption of resource \(r\) per unit of product \(i\) | `file_0_view_0.r1_per_unit`, `r2_per_unit`, `r3_per_unit` |
| \(U_i\) | Upper demand (units) for product \(i\) | `file_0_view_0.upper_demand_units` |
| \(C_r\) | Total capacity of resource \(r\) | `file_1_view_0.capacity` |

Decision variables  

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

( \(x_i\) = number of 10‑unit batches produced of product \(i\) )

---

**Objective**

\[
\max \;\; \sum_{i\in I} 10\,p_i\,x_i
\]

---

**Constraints**

1. **Resource limits**

\[
\sum_{i\in I} 10\,a_{ri}\,x_i \;\le\; C_r \qquad \forall r\in R
\]

2. **Demand limits**

\[
10\,x_i \;\le\; U_i \qquad \forall i\in I
\]

3. **Non‑negativity & integrality**

\[
x_i \in \mathbb{Z}_{\ge 0}\qquad \forall i\in I
\]

---

**Data Mapping Rules**

| Parameter | Mapping expression (CSVQA_DATA) |
|-----------|---------------------------------|
| \(p_i\) | `file_0_view_0.profit_per_unit` where `product = i` |
| \(a_{1i}\) | `file_0_view_0.r1_per_unit` where `product = i` |
| \(a_{2i}\) | `file_0_view_0.r2_per_unit` where `product = i` |
| \(a_{3i}\) | `file_0_view_0.r3_per_unit` where `product = i` |
| \(U_i\) | `file_0_view_0.upper_demand_units` where `product = i` |
| \(C_{R1}\) | `file_1_view_0.capacity` where `resource = 'R1'` |
| \(C_{R2}\) | `file_1_view_0.capacity` where `resource = 'R2'` |
| \(C_{R3}\) | `file_1_view_0.capacity` where `resource = 'R3'` |

All indices and parameters are derived directly from the complete runtime records; no values are enumerated in the model.