## Mathematical Model

**Sets**  
\(I=\{1,2,3\}\) – CPUs (1: 1.33 GHz, 2: 2 GHz, 3: 2.66 GHz)  
\(J=\{1,\dots ,40\}\) – tasks  

**Parameters**  
\(BI_j\) – basic instructions of task \(j\) (billions of instructions)  
\(f_i\) – frequency of CPU \(i\) (GHz)  
\(p_{ij}=BI_j/f_i\) – processing time of task \(j\) on CPU \(i\) (seconds)

*Data mapping*  
- Table **file_0_view_0** (column “Process” row “BI”) supplies \(BI_j\) for each \(j\).  
- Frequencies are fixed constants:  
  \(f_1=1.33,\; f_2=2,\; f_3=2.66\).  
- For every \(i\in I,\; j\in J\) compute \(p_{ij}=BI_j/f_i\).

**Decision Variables**  
\(x_{ij}\in\{0,1\}\) – 1 if task \(j\) is assigned to CPU \(i\).  
\(C_i\ge 0\) – completion time of CPU \(i\).  
\(C_{\max}\ge 0\) – makespan (completion time of the last task).

**Constraints**

1. **Task assignment**  
\[
\sum_{i\in I} x_{ij}=1 \qquad \forall j\in J
\]

2. **CPU load**  
\[
C_i \;\ge\; \sum_{j\in J} p_{ij}\,x_{ij} \qquad \forall i\in I
\]

3. **Makespan definition**  
\[
C_{\max}\;\ge\; C_i \qquad \forall i\in I
\]

**Objective**

\[
\min \; C_{\max}
\]

---

### Data Mapping Rules

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(BI_j\) | `file_0_view_0` – column “Process”, row “BI” | For each task \(j\), read the value in column \(j\). |
| \(f_i\) | Fixed in problem statement | \(f_1=1.33,\; f_2=2,\; f_3=2.66\). |
| \(p_{ij}\) | Derived | \(p_{ij}=BI_j/f_i\) for all \(i\in I,\; j\in J\). |

All other parameters are constants defined above. No additional data or constraints are required.