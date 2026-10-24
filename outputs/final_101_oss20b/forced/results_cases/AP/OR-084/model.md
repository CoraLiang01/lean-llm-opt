**Mathematical Model**

```markdown
##### Sets
- $M = \{1,2,3\}$  (3 CPUs)  
- $T = \{1,2,\dots ,40\}$  (40 tasks)

##### Parameters
- $BI_j \;\; (j\in T)$ – basic instructions (billions) for task $j$  
- $f_i \;\; (i\in M)$ – frequency of CPU $i$ (GHz)  
  \[
  f_1 = 1.33,\quad f_2 = 2,\quad f_3 = 2.66
  \]

- Derived processing time  
  \[
  p_{ij} = \frac{BI_j}{f_i}\qquad (i\in M,\; j\in T)
  \]

##### Decision Variables
- $x_{ij}\in\{0,1\}$ = 1 if task $j$ is assigned to CPU $i$, 0 otherwise  
- $C_{\max}\ge 0$                               