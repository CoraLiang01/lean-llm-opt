**Mathematical Model**

Let  

* \(T=\{1,\dots ,40\}\) – set of tasks  
* \(C=\{1,2,3\}\) – set of CPUs  

Parameters  

* \(\text{BI}_i\) – basic instructions (billions) for task \(i\) (from *18.csv*)  
* \(f_c\) – CPU frequency (GHz)  
  \[
  f_1=1.33,\qquad f_2=2,\qquad f_3=2.66
  \]
* Derived processing time  
  \[
  t_{i,c}=\frac{\text{BI}_i}{f_c}\quad\forall i\in T,\;c\in C
  \]

Decision variables  

* \(x_{i,c}\in\{0,1\}\) – 1 if task \(i\) is assigned to CPU \(c\)  
* \(M\ge 0\) – makespan (completion time of the last task)

**Objective**

\[
\min \; M
\]

**Constraints**

1. **Task assignment**  
   \[
   \sum_{c\in C} x_{i,c}=1 \quad \forall i\in T
   \]

2. **CPU workload (makespan)**  
   \[
   \sum_{i\in T} t_{i,c}\,x_{i,c}\;\le\; M \quad \forall c\in C
   \]

---

### Data Mapping

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(\text{BI}_i\) | Basic instructions for each task | `file_0_view_0` | Columns `"1"` through `"40"` (each column holds the BI value for task \(i\)) |
| \(f_c\) | CPU frequencies (given in query) | – | – |
| \(t_{i,c}\) | Derived | – | – |

*The table `file_0_view_0` contains a single row with columns `"Process"` and `"1"`–`"40"`. The BI values are taken from the columns `"1"`–`"40"`.*