**Mathematical Model**

Let  

* \(T=\{1,\dots ,40\}\) – set of tasks  
* \(M=\{1,2,3\}\) – set of CPUs  
* \(f_1=1.33,\; f_2=2,\; f_3=2.66\) – CPU frequencies (GHz)  

Parameters  

* \(BI_j\) – basic instructions of task \(j\) (billions of instructions)  
* \(p_{ij}=BI_j/f_i\) – processing time of task \(j\) on CPU \(i\) (seconds)  

Decision variables  

* \(x_{ij}\in\{0,1\}\) – 1 if task \(j\) is assigned to CPU \(i\)  
* \(s_j\ge 0\) – start time of task \(j\)  
* \(C_j=s_j+p_{ij}\) – completion time of task \(j\)  
* \(C_{\max}\ge 0\) – makespan (completion time of the last task)  
* \(y_{jk}\in\{0,1\}\) – 1 if task \(j\) precedes task \(k\) on the same CPU  

Big‑\(M\) constant \(M\) is any number larger than the sum of all processing times.  

---

### Objective  

\[
\min \; C_{\max}
\]

### Constraints  

1. **Assignment**  
\[
\sum_{i\in M} x_{ij}=1 \quad \forall j\in T
\]

2. **Start–completion relation**  
\[
C_j = s_j + \sum_{i\in M} p_{ij}\,x_{ij} \quad \forall j\in T
\]

3. **CPU sequencing (disjunctive constraints)**  
For every unordered pair \(\{j,k\}\subset T,\; j<k\):

\[
\begin{aligned}
s_k &\ge C_j - M\,(1-y_{jk}) \\
s_j &\ge C_k - M\,y_{jk}
\end{aligned}
\qquad \forall j<k
\]

4. **Makespan definition**  
\[
C_j \le C_{\max} \quad \forall j\in T
\]

5. **Variable domains**  

\[
x_{ij}\in\{0,1\},\; y_{jk}\in\{0,1\},\; s_j\ge 0,\; C_j\ge 0,\; C_{\max}\ge 0
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| `BI_j` | `file_0_view_0` | Column `"1"` → `BI_1`, `"2"` → `BI_2`, … , `"40"` → `BI_40` |
| `p_{ij}` | Derived | \(p_{ij}=BI_j/f_i\) for each \(i\in M,\; j\in T\) |

*Table ID:* `file_0_view_0`  
*Columns used:* `"1"`, `"2"`, … , `"40"` (all task instruction counts).  

No numerical values are enumerated; the model references the data symbolically.