**Symbolic Linear‑Programming Model**

Let  

* \(T=\{1,2,\dots ,48\}\) – indices of the 30‑minute periods in the 24‑h day  
* \(x_t\in\{0,1\}\) – 1 if a waitstaff starts a shift at period \(t\), 0 otherwise  
* \(r_t\) – required number of waitstaff during period \(t\) (parameter)  

Define the coverage indicator  

\[
c_{t,p}= 
\begin{cases}
1 & \text{if a shift that starts at }t\text{ covers period }p\\
0 & \text{otherwise}
\end{cases}
=
\begin{cases}
1 & \text{if }(p-t+48)\bmod 48 < 16\\
0 & \text{otherwise}
\end{cases}
\]

(Each shift lasts 8 h = 16 half‑hour periods, with wrap‑around at midnight.)

---

### Decision Variables
\[
x_t \in \{0,1\}\quad \forall t\in T
\]

### Objective
\[
\min \sum_{t\in T} x_t
\]

### Constraints
For every period \(p\in T\) the number of active shifts must meet the requirement:

\[
\sum_{t\in T} c_{t,p}\,x_t \;\ge\; r_p \qquad \forall p\in T
\]

---

## Data Mapping Rules

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(T\) | Implicit (period indices) | \(T=\{1,\dots ,48\}\) |
| \(r_t\) | `file_0_view_0` (column **Requirement**) | For each row \(k\) (0‑based) in the table, set \(r_{k+1}=\text{Requirement}_k\). |
| \(c_{t,p}\) | Derived | \(c_{t,p}=1\) if \((p-t+48)\bmod 48 < 16\); otherwise \(0\). |

*Table `file_0_view_0`*  
- **Columns**: `Time`, `Requirement`  
- **Rows**: 48 rows, each representing a consecutive 30‑minute interval.  
- **Requirement** values are integer counts of waitstaff needed for that interval.

No other data are required. The model is a binary set‑covering integer program that yields the minimum number of waitstaff needed to satisfy all hourly requirements.