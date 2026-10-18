**Abstract Linear Programming Model**

**Index Set**  
\(T = \{1,2,\dots ,24\}\) (shift indices)

**Parameters**  
\(r_t \; (t \in T)\) – required number of drivers/crew members for shift \(t\)  
\(r_t\) is obtained from the column **Number Required** of the table **file_0_view_0** (CSV file 42.csv).

**Decision Variables**  
\(x_t \in \mathbb{Z}_{\ge 0}\) (number of drivers/crew members who start working at the beginning of shift \(t\))

**Objective Function**  
\[
\quad \quad \min \quad \sum_{t \in T} x_t
\]

**Constraints**

1. **Coverage Constraints** – each shift must be covered by drivers/crew who started in the current or previous three shifts  
\[
\quad \quad \sum_{k=\max\{1,t-3\}}^{t} x_k \;\ge\; r_t , \qquad \forall\, t \in T
\]

2. **Non‑negativity and Integer Requirement**  
\[
\quad \quad x_t \in \mathbb{Z}_{\ge 0}, \qquad \forall\, t \in T
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(r_t\)   | file_0_view_0 (42.csv) | Number Required |

The table **file_0_view_0** contains 24 rows, each row corresponding to a shift \(t\) with columns **Shift**, **Time**, and **Number Required**. The value of \(r_t\) is taken directly from the **Number Required** column for each shift \(t\).