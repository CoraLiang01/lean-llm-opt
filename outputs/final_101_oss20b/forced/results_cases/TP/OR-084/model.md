**Mathematical Model**

Let  

* \(I=\{1,2,3\}\) – set of CPUs (processor 1: 1.33 GHz, processor 2: 2 GHz, processor 3: 2.66 GHz)  
* \(J=\{1,\dots ,40\}\) – set of tasks  

Parameters  

* \(f_i\) – frequency of CPU \(i\)  
  \[
  f_1=1.33,\; f_2=2,\; f_3=2.66
  \]
* \(b_j\) – basic instructions (in billions) of task \(j\) (from *18.csv*)  
* \(p_{ij}=b_j/f_i\) – processing time of task \(j\) on CPU \(i\)

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if task \(j\) is assigned to CPU \(i\)  
* \(C_i\ge 0\) – completion time (load) of CPU \(i\)  
* \(C_{\max}\ge 0\) – makespan (maximum completion time)

Objective  

\[
\min\; C_{\max}
\]

Constraints  

1. **Load definition**  
   \[
   C_i=\sum_{j\in J} p_{ij}\,x_{ij}\qquad \forall i\in I
   \]

2. **Makespan upper bound**  
   \[
   C_i \le C_{\max}\qquad \forall i\in I
   \]

3. **Task assignment**  
   \[
   \sum_{i\in I} x_{ij}=1\qquad \forall j\in J
   \]

4. **Binary assignment**  
   \[
   x_{ij}\in\{0,1\}\qquad \forall i\in I,\; j\in J
   \]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(f_i\) | Fixed in model | \(f_1=1.33,\; f_2=2,\; f_3=2.66\) |
| \(b_j\) | *18.csv* (table_id `file_0_view_0`) | Column `Process` is the row label; columns `1`–`40` contain \(b_j\) values. |
| \(p_{ij}\) | Derived | \(p_{ij}=b_j/f_i\) for all \(i\in I,\; j\in J\) |

*Indices*  

* \(I=\{1,2,3\}\) – CPU identifiers (matching the frequency list).  
* \(J=\{1,\dots ,40\}\) – task identifiers (matching the column names in *18.csv*).  

All other data (e.g., frequencies) are constants defined in the model. No additional constraints are required.