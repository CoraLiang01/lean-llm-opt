**Mathematical Model**

Let  

- \(M=\{M1,\dots ,M12\}\) (index set of machines)  
- \(T=\{A,B,C,D,E,F,G,H,I,J,K,L\}\) (index set of tasks)  

Decision variable  
\[
x_{m t}\in\{0,1\}\quad\forall m\in M,\;t\in T
\]
(1 if machine \(m\) is assigned to task \(t\), 0 otherwise).

Parameter  
\[
c_{m t}\ge 0 \quad\text{(machining cost of assigning machine }m\text{ to task }t)
\]

Objective  
\[
\min \sum_{m\in M}\sum_{t\in T} c_{m t}\,x_{m t}
\]

Constraints  

1. **Machine assignment**  
\[
\sum_{t\in T} x_{m t}=1 \qquad \forall m\in M
\]

2. **Task assignment**  
\[
\sum_{m\in M} x_{m t}=1 \qquad \forall t\in T
\]

---

**Data Mapping Rules**

| Parameter | Source | Table ID | Column Mapping |
|-----------|--------|----------|----------------|
| \(c_{m t}\) | Cost matrix | `file_0_view_0` | `Machine` → \(m\); columns `A`–`L` → \(t\) |

All other data (index sets, variable domains, objective sense, and constraints) are defined by the model above and do not require additional tables.