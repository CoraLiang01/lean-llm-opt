##### Decision Variables

$x_{ij}\geq0$: processing time of task $j$ on CPU $i$ (continuous).  
$y_{ij}\in\{0,1\}$: $1$ if task $j$ is assigned to CPU $i$, $0$ otherwise.  
$C_{\max}\geq0$: makespan (completion time of the last task).

##### Objective Function

$\min \; C_{\max}$

##### Constraints

1. **Task assignment**  
$\displaystyle \sum_{i=1}^{3} y_{ij}=1,\quad \forall j\in\{1,\dots ,40\}$  

2. **Processing time definition**  
$\displaystyle x_{ij}=\frac{BI_j}{s_i}\,y_{ij},\quad \forall i\in\{1,2,3\},\;\forall j\in\{1,\dots ,40\}$  

3. **CPU load**  
$\displaystyle \sum_{j=1}^{40} x_{ij}\leq C_{\max},\quad \forall i\in\{1,2,3\}$  

4. **Domains**  
$y_{ij}\in\{0,1\},\;x_{ij}\geq0,\;C_{\max}\geq0$  

$BI_j$ is the basic instruction count (in billions) for task $j$.  
$s_i$ is the frequency (in GHz) of CPU $i$.

##### Parameters

$BI=\bigl[1.1,\;2.1,\;3,\;1,\;0.7,\;5,\;3,\;3.5,\;4.4,\;3.8,\;3.5,\;2.8,\;4.1,\;2.9,\;5.4,\;5.8,\;2.6,\;4.9,\;3.4,\;3.6,\;5.6,\;0.9,\;1,\;0.6,\;5.1,\;4.8,\;5.3,\;5.9,\;4.9,\;3,\;4.8,\;1.2,\;4,\;1.3,\;5.7,\;3.4,\;2.8,\;2,\;4.8,\;3\bigr]$

$s=\bigl[1.33,\;2,\;2.66\bigr]$

$M=\sum_{j=1}^{40} BI_j= \displaystyle 1.1+2.1+3+1+0.7+5+3+3.5+4.4+3.8+3.5+2.8+4.1+2.9+5.4+5.8+2.6+4.9+3.4+3.6+5.6+0.9+1+0.6+5.1+4.8+5.3+5.9+4.9+3+4.8+1.2+4+1.3+5.7+3.4+2.8+2+4.8+3= \displaystyle 107.6$

###### Retrieved Information

{
  "tasks": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
  "frequencies_GHz": [1.33, 2, 2.66],
  "BI_vector_BI": [1.1, 2.1, 3, 1, 0.7, 5, 3, 3.5, 4.4, 3.8, 3.5, 2.8, 4.1, 2.9, 5.4, 5.8, 2.6, 4.9, 3.4, 3.6, 5.6, 0.9, 1, 0.6, 5.1, 4.8, 5.3, 5.9, 4.9, 3, 4.8, 1.2, 4, 1.3, 5.7, 3.4, 2.8, 2, 4.8, 3]
}