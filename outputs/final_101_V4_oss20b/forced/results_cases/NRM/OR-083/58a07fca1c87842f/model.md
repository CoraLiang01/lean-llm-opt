**Symbolic Model**

\[
\begin{aligned}
\text{Sets} \quad & W=\{1,\dots ,12\} \quad &&\text{(workers)}\\
& T=\{A,B,\dots ,J\} \quad &&\text{(tasks)}\\[4pt]

\text{Parameters} \quad & t_{i,j}\quad &&\text{time required for worker }i\text{ to perform task }j\\[4pt]

\text{Decision Variables} \quad & x_{i,j}\in\{0,1\}\quad &&\text{1 if worker }i\text{ is assigned to task }j\\[4pt]

\text{Objective} \quad & \min \;\sum_{i\in W}\sum_{j\in T} t_{i,j}\,x_{i,j}\\[4pt]

\text{Constraints} \\
& \sum_{i\in W} x_{i,j}=1 \qquad &&\forall j\in T \quad\text{(each task gets one worker)}\\
& \sum_{j\in T} x_{i,j}\le 1 \qquad &&\forall i\in W \quad\text{(each worker does at most one task)}\\
& \sum_{i\in W}\sum_{j\in T} x_{i,j}=10 \qquad &&\text{(exactly ten workers are used)}\\
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(t_{i,j}\) | `file_0_view_0` | For each row where **Task Time Required** = \(i\) (worker index), read the value in column **\(j\)** (task A–J). That value is \(t_{i,j}\). |

*No numeric values are listed; the model references the table and columns exactly as they appear in the source.*