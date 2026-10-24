**Mathematical Model**

Let  

* \(M=\{M1,M2,M3,M4\}\) – set of technician teams (machines).  
* \(J=\{J1,\dots ,J8\}\) – set of jobs.  

Parameters  

* \(c_{ij}\) – assignment cost of job \(j\) to team \(i\).  
* \(r_{ij}\) – capacity consumed on team \(i\) when job \(j\) is assigned.  
* \(C_i\) – available capacity of team \(i\).  

Decision variables  

\[
x_{ij}\in\{0,1\}\quad \forall i\in M,\; j\in J
\]

---

**Objective**

\[
\min \;\sum_{i\in M}\sum_{j\in J} c_{ij}\,x_{ij}
\]

---

**Constraints**

1. **Job‑assignment** – each job is assigned to exactly one team  

\[
\sum_{i\in M} x_{ij}=1 \qquad \forall j\in J
\]

2. **Capacity** – capacity used on each team does not exceed its limit  

\[
\sum_{j\in J} r_{ij}\,x_{ij}\;\le\; C_i \qquad \forall i\in M
\]

3. **Binary** – already encoded in the variable definition  

\[
x_{ij}\in\{0,1\}\qquad \forall i\in M,\; j\in J
\]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(c_{ij}\) | `file_1_view_0` | `J1`–`J8` (job columns) |
| \(r_{ij}\) | `file_2_view_0` | `J1`–`J8` (job columns) |

*The index `Machine` in each table provides the key \(i\).  
The job columns `J1` … `J8` provide the key \(j\).*

All parameters are taken directly from the corresponding CSV views; no values are hard‑coded in the model.