**Mathematical Model**

Let  

* \(I=\{1,\dots ,40\}\) – set of tasks  
* \(J=\{1,2,3\}\) – set of CPUs  

Parameters  

* \(BI_i\) – basic instructions (billions) of task \(i\)  
* \(f_j\) – frequency of CPU \(j\) (GHz)  
* \(P_{ij}=BI_i/f_j\) – processing time of task \(i\) on CPU \(j\) (seconds)  

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if task \(i\) is assigned to CPU \(j\)  
* \(s_i\ge 0\) – start time of task \(i\)  
* \(C_i\ge 0\) – completion time of task \(i\)  
* \(z_{ikj}\in\{0,1\}\) – 1 if task \(i\) precedes task \(k\) on CPU \(j\) (for \(i<k\))  
* \(T\ge 0\) – makespan (completion time of the last task)

---

### Objective
\[
\min \; T
\]

### Constraints

1. **Assignment**  
\[
\sum_{j\in J} x_{ij}=1 \qquad \forall i\in I
\]

2. **Completion time**  
\[
C_i \;\ge\; s_i + \sum_{j\in J} P_{ij}\,x_{ij} \qquad \forall i\in I
\]

3. **Makespan**  
\[
T \;\ge\; C_i \qquad \forall i\in I
\]

4. **Non‑overlap on the same CPU**  
Let \(M\) be a sufficiently large constant (e.g. \(M=\sum_{i\in I}\max_{j\in J}P_{ij}\)).  
For every pair \(i<k\) and every CPU \(j\):

\[
\begin{aligned}
s_k &\;\ge\; s_i + P_{ij} - M\,(1-z_{ikj}) - M\,(1-x_{ij}) - M\,(1-x_{kj}) \\
s_i &\;\ge\; s_k + P_{kj} - M\,z_{ikj} - M\,(1-x_{ij}) - M\,(1-x_{kj}) \\
z_{ikj} + z_{kij} &= 1 \\
z_{ikj} &\le x_{ij} \\
z_{ikj} &\le x_{kj}
\end{aligned}
\qquad \forall i<k,\; j\in J
\]

5. **Variable domains**  
\[
x_{ij}\in\{0,1\},\; z_{ikj}\in\{0,1\},\; s_i\ge0,\; C_i\ge0,\; T\ge0
\]

---

## Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(BI_i\) | `file_0_view_0` | Column *i* (where `Process="BI"`) |
| \(f_1\) | Problem statement | 1.33 |
| \(f_2\) | Problem statement | 2.00 |
| \(f_3\) | Problem statement | 2.66 |
| \(P_{ij}\) | Derived | \(P_{ij}=BI_i/f_j\) |

All other constants (e.g., \(M\)) are to be chosen large enough to render the big‑\(M\) constraints non‑binding when the corresponding tasks are not on the same CPU.