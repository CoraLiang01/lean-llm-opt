**Symbolic Model**

Let  

* \(J=\{1,\dots ,40\}\) – set of tasks  
* \(K=\{1,2,3\}\) – set of CPUs (processor 1: 1.33 GHz, processor 2: 2 GHz, processor 3: 2.66 GHz)

Parameters  

* \(BI_j\) – basic instructions (billions) of task \(j\) (from *18.csv*)  
* \(f_k\) – frequency of processor \(k\) (GHz)  
* \(p_{jk}=BI_j/f_k\) – processing time of task \(j\) on processor \(k\) (seconds)

Decision variables  

* \(x_{jk}\in\{0,1\}\) – 1 if task \(j\) is assigned to processor \(k\)  
* \(C_{\max}\ge 0\) – makespan (completion time of the last task)

---

**Objective**

\[
\min \; C_{\max}
\]

---

**Constraints**

1. **Task assignment**  
   \[
   \sum_{k\in K} x_{jk}=1 \qquad \forall j\in J
   \]

2. **Processor load**  
   \[
   \sum_{j\in J} p_{jk}\,x_{jk}\;\le\; C_{\max} \qquad \forall k\in K
   \]

3. **Variable domains**  
   \[
   x_{jk}\in\{0,1\}\;\;\forall j\in J,\;k\in K,\qquad
   C_{\max}\ge 0
   \]

---

**Data Mapping Rules**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(BI_j\) | *18.csv* (table `file_0_view_0`) | `BI_j = value in column j of the single row where "Process" = "BI"` |
| \(f_1\) | Problem statement | `f_1 = 1.33` |
| \(f_2\) | Problem statement | `f_2 = 2.0` |
| \(f_3\) | Problem statement | `f_3 = 2.66` |
| \(p_{jk}\) | Derived | `p_{jk} = BI_j / f_k` |

No other data are required. The model uses only the task basic‑instruction values from *18.csv* and the fixed processor frequencies.