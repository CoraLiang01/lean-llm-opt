**Mathematical Model**

Let  

- \(M=\{M1,\dots ,M12\}\) (index set of machines)  
- \(T=\{A,B,\dots ,L\}\) (index set of tasks)  

Decision variable  

\[
x_{m t}\in\{0,1\}\qquad \forall m\in M,\;t\in T
\]

Parameter  

\[
c_{m t}\ge 0 \qquad \forall m\in M,\;t\in T
\]

Objective (minimum‑cost assignment)

\[
\min \;\sum_{m\in M}\sum_{t\in T} c_{m t}\,x_{m t}
\]

Constraints  

1. **Machine‑to‑task** (each machine to exactly one task)  

\[
\sum_{t\in T} x_{m t}=1 \qquad \forall m\in M
\]

2. **Task‑to‑machine** (each task to exactly one machine)  

\[
\sum_{m\in M} x_{m t}=1 \qquad \forall t\in T
\]

---

**Data Mapping Rules**

| Symbol | Source | Mapping Details |
|--------|--------|-----------------|
| \(M\) | `cost_12x12.csv` – column **Machine** | Each distinct value in the **Machine** column (rows 0–11) defines an element of \(M\). |
| \(T\) | `cost_12x12.csv` – columns **A**–**L** | Each column header **A** through **L** defines an element of \(T\). |
| \(c_{m t}\) | `cost_12x12.csv` – cell at row *m*, column *t* | For each machine row and task column, the numeric value in that cell is the cost \(c_{m t}\). |
| \(x_{m t}\) | Decision variable | Binary variable indexed by \(m\in M,\;t\in T\). |
| Objective sense | Query | Minimize total cost. |
| Constraints | Query | Equality constraints as listed above. |

All data are taken directly from the table `file_0_view_0` (the view of `cost_12x12.csv`). No additional parameters or constraints are introduced.