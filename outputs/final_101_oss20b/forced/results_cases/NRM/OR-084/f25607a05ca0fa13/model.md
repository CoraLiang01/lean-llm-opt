##### Abstract Model

**Index Sets**  
$M = \{1,2,3\}$ (CPUs)  
$J = \{1,2,\dots ,40\}$ (Tasks)

**Parameters**  
$BI_j$ – basic instructions (billions) for task $j$  
$f_i$ – frequency (GHz) of CPU $i$  
$p_{ij} = \dfrac{BI_j}{f_i}$ – processing time of task $j$ on CPU $i$

**Decision Variables**  
$x_{ij} \in \{0,1\}$ – equals 1 if task $j$ is assigned to CPU $i$, 0 otherwise  
$C_{\max} \in \mathbb{R}_{+}$ – makespan (completion time of the last task)

**Objective**  
$\displaystyle \min \; C_{\max}$

**Constraints**

1. **Task Assignment**  
$\displaystyle \sum_{i\in M} x_{ij} = 1, \quad \forall j\in J$

2. **Machine Load**  
$\displaystyle \sum_{j\in J} p_{ij}\,x_{ij} \;\le\; C_{\max}, \quad \forall i\in M$

3. **Variable Domains**  
$\displaystyle x_{ij}\in\{0,1\}, \quad \forall i\in M,\; j\in J$  
$\displaystyle C_{\max}\ge 0$

---

##### Data Mapping

| Parameter | Source | Table ID | Column |
|-----------|--------|----------|--------|
| $BI_j$ | Basic instructions for each task | `file_0_view_0` | Columns `"1"` through `"40"` |
| $f_1$ | CPU 1 frequency | — | 1.33 GHz |
| $f_2$ | CPU 2 frequency | — | 2.00 GHz |
| $f_3$ | CPU 3 frequency | — | 2.66 GHz |

*The processing time parameter $p_{ij}$ is derived from $BI_j$ and $f_i$ as shown above.*