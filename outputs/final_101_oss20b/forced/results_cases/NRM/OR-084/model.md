**Mathematical Model**

Let  

* \(T=\{1,\dots ,40\}\) – set of tasks  
* \(P=\{1,2,3\}\) – set of processors  
* \(f_p\) – frequency of processor \(p\) (GHz)  
  \[
  f_1=1.33,\qquad f_2=2,\qquad f_3=2.66
  \]
* \(b_t\) – basic instructions of task \(t\) (billions of instructions, BI)  

Decision variables  

* \(x_{t,p}\in\{0,1\}\) – 1 if task \(t\) is assigned to processor \(p\)  
* \(C_{\max}\ge 0\) – makespan (completion time of the last task)

---

### Objective
\[
\min \; C_{\max}
\]

### Constraints
1. **Task assignment**  
   \[
   \sum_{p\in P} x_{t,p}=1 \qquad \forall t\in T
   \]

2. **Processor load (makespan bound)**  
   \[
   \sum_{t\in T}\frac{b_t}{f_p}\,x_{t,p}\;\le\; C_{\max}\qquad \forall p\in P
   \]

3. **Binary assignment**  
   \[
   x_{t,p}\in\{0,1\}\qquad \forall t\in T,\;p\in P
   \]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(b_t\) | Basic instructions for each task | `file_0_view_0` | Columns `"1"`–`"40"` in the row where `"Process"="BI"` |
| \(f_1,f_2,f_3\) | Processor frequencies | *query‑defined* | constants 1.33, 2, 2.66 |

All other data (task indices, processor indices) are defined by the model itself. No additional parameters are required.