**Mathematical Model**

Let  

* \(I=\{1,2,3\}\) – set of CPUs (processor 1: 1.33 GHz, processor 2: 2 GHz, processor 3: 2.66 GHz)  
* \(J=\{1,\dots ,40\}\) – set of tasks  

Parameters  

* \(f_i\) – frequency of processor \(i\) (GHz)  
  \[
  f_1=1.33,\qquad f_2=2,\qquad f_3=2.66
  \]
* \(BI_j\) – basic instructions of task \(j\) (billions) – taken from **18.csv**  
* \(t_{ij}=BI_j/f_i\) – processing time of task \(j\) on processor \(i\) (seconds)

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if task \(j\) is assigned to processor \(i\), 0 otherwise  
* \(C_i\ge 0\) – completion time of processor \(i\)  
* \(C_{\max}\ge 0\) – makespan (completion time of the last finishing processor)

Objective  

\[
\min \; C_{\max}
\]

Constraints  

1. **Task assignment**  
   \[
   \sum_{i\in I} x_{ij}=1,\qquad \forall j\in J
   \]

2. **Processor completion time**  
   \[
   C_i=\sum_{j\in J} t_{ij}\,x_{ij},\qquad \forall i\in I
   \]

3. **Makespan definition**  
   \[
   C_{\max}\ge C_i,\qquad \forall i\in I
   \]

4. **Variable domains**  
   \[
   x_{ij}\in\{0,1\},\; C_i\ge 0,\; C_{\max}\ge 0
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(BI_j\) | **18.csv** (table_id `file_0_view_0`) | `BI_j` = value in column `j` of the row where `Process` = `"BI"` |
| \(f_1\) | Fixed | `f_1 = 1.33` |
| \(f_2\) | Fixed | `f_2 = 2` |
| \(f_3\) | Fixed | `f_3 = 2.66` |
| \(t_{ij}\) | Derived | `t_{ij} = BI_j / f_i` |

*Index sets*  
- `I = {1, 2, 3}`  
- `J = {1, 2, …, 40}`

*Note*: No other data are required; all numerical values are obtained from the mapping above.