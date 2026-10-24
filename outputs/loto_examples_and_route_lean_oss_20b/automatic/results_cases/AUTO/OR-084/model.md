**Mathematical Model**

Let  

* \(T=\{1,\dots ,40\}\) – set of tasks  
* \(K=\{1,2,3\}\) – set of CPUs  

Parameters  

* \(BI_i\) – basic instructions (billions) of task \(i\) (from *file_0_view_0*)  
* \(f_k\) – frequency of CPU \(k\)  
  \[
  f_1=1.33,\; f_2=2,\; f_3=2.66
  \]
* \(p_{ik}=BI_i/f_k\) – processing time of task \(i\) on CPU \(k\)

Variables  

* \(y_{ik}\in\{0,1\}\) – 1 if task \(i\) is assigned to CPU \(k\)  
* \(s_i\ge 0\) – start time of task \(i\)  
* \(C_{\max}\ge 0\) – makespan (completion time of the last task)  
* \(z_{ij}^k\in\{0,1\}\) – 1 if task \(i\) precedes task \(j\) on CPU \(k\) (for all \(i<j\))

---

### Objective
\[
\min \; C_{\max}
\]

### Constraints  

1. **Assignment**  
   \[
   \sum_{k\in K} y_{ik}=1 \qquad \forall i\in T
   \]

2. **Sequencing on each CPU**  
   For every pair \(i<j\) and CPU \(k\):

   \[
   \begin{aligned}
   s_i + p_{ik} &\le s_j + M\,(1-z_{ij}^k) + M\,(2-y_{ik}-y_{jk}) \\
   s_j + p_{jk} &\le s_i + M\,z_{ij}^k + M\,(2-y_{ik}-y_{jk}) \\
   z_{ij}^k + z_{ji}^k &\le 1 \\
   z_{ij}^k + z_{ji}^k &\ge y_{ik}+y_{jk}-1
   \end{aligned}
   \]
   (Here \(z_{ji}^k\) is the complementary ordering variable; \(M\) is a sufficiently large constant, e.g. the sum of all \(p_{ik}\).)

3. **Makespan definition**  
   \[
   C_{\max} \ge s_i + p_{ik} - M\,(1-y_{ik}) \qquad \forall i\in T,\;k\in K
   \]

4. **Variable domains**  
   \[
   y_{ik}\in\{0,1\},\; z_{ij}^k\in\{0,1\},\; s_i\ge 0,\; C_{\max}\ge 0
   \]

---

## Data Mapping

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(BI_i\) | `file_0_view_0` | `BI[i] = value of column i in the single row` |
| \(f_k\) | Constant | `f[1]=1.33, f[2]=2, f[3]=2.66` |
| \(p_{ik}\) | Derived | `p[i,k] = BI[i] / f[k]` |

All other parameters are defined by the model equations above. No additional data are required.