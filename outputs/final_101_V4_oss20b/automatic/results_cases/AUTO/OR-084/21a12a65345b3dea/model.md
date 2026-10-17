## Mathematical Model

**Indices**  
\(T=\{1,\dots ,40\}\) – tasks  
\(M=\{1,2,3\}\) – CPUs (1:1.33 GHz, 2:2 GHz, 3:2.66 GHz)  
\(P=\{(i,k,j)\mid i,k\in T,\;i<k,\;j\in M\}\) – ordered task pairs on each CPU  

**Parameters**  
\(\text{BI}_i\) – basic instructions (billions) for task \(i\) (from *18.csv*, column \(i\))  
\(f_j\) – frequency of CPU \(j\) (1.33, 2, 2.66)  
\(p_{ij}= \dfrac{\text{BI}_i}{f_j}\) – processing time of task \(i\) on CPU \(j\)  
\(M\) – a sufficiently large constant (e.g. \(\sum_{i\in T}\max_j p_{ij}\))

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) – 1 if task \(i\) is assigned to CPU \(j\)  
\(y_{ikj}\in\{0,1\}\) – 1 if task \(i\) precedes task \(k\) on CPU \(j\)  
\(s_i\ge 0\) – start time of task \(i\)  
\(c_i\ge 0\) – completion time of task \(i\)  
\(C_{\max}\ge 0\) – makespan (completion time of the last task)

**Constraints**

1. **Assignment**  
\[
\sum_{j\in M} x_{ij}=1 \qquad \forall i\in T
\]

2. **Ordering variables**  
\[
\begin{aligned}
y_{ikj} &\le x_{ij} \\
y_{ikj} &\le x_{kj} \\
y_{ikj} &\ge x_{ij}+x_{kj}-1
\end{aligned}
\qquad \forall (i,k,j)\in P
\]

3. **Disjunctive sequencing**  
\[
\begin{aligned}
s_i + p_{ij} &\le s_k + M\,(1-y_{ikj}) \\
s_k + p_{kj} &\le s_i + M\,y_{ikj}
\end{aligned}
\qquad \forall (i,k,j)\in P
\]

4. **Completion time definition**  
\[
c_i = s_i + \sum_{j\in M} p_{ij}\,x_{ij} \qquad \forall i\in T
\]

5. **Makespan**  
\[
c_i \le C_{\max} \qquad \forall i\in T
\]

**Objective**  
\[
\min \; C_{\max}
\]

---

## Data Mapping Rules

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(\text{BI}_i\) | *18.csv* (table_id `file_0_view_0`) | Column `i` of the single row (values 1.1, 2.1, …, 4.8) |
| \(f_1\) | Constant | 1.33 |
| \(f_2\) | Constant | 2.00 |
| \(f_3\) | Constant | 2.66 |
| \(p_{ij}\) | Derived | \(\text{BI}_i / f_j\) |
| \(M\) | Derived | Any value ≥ \(\sum_{i\in T}\max_j p_{ij}\) |

All other indices, variables, and constraints are defined solely by the model and the above data mapping. No additional data or assumptions are introduced.