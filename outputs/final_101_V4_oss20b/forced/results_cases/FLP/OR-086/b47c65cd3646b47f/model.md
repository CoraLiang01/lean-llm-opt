##### Decision Variables

$x_{ij}\geq0$: quantity of raw grade $i\in\{I,II,III\}$ used to produce brand $j\in\{Red,Yellow,Blue\}$ (continuous).  

##### Objective Function

$\displaystyle \max \;\sum_{j\in\{Red,Yellow,Blue\}}p_j\sum_{i\in\{I,II,III\}}x_{ij}\;-\;\sum_{i\in\{I,II,III\}}c_i\sum_{j\in\{Red,Yellow,Blue\}}x_{ij}$  

##### Constraints  

1. **Raw material supply limits**  
   $\displaystyle \sum_{j\in\{Red,Yellow,Blue\}}x_{ij}\;\leq\;d_i,\quad \forall i\in\{I,II,III\}$  

2. **Minimum production for Red**  
   $\displaystyle \sum_{i\in\{I,II,III\}}x_{i,Red}\;\geq\;2000$  

3. **Blending lower‑bound proportions**  
   $\displaystyle l_{ij}\,\sum_{k\in\{I,II,III\}}x_{kj}\;\leq\;x_{ij},\quad \forall i\in\{I,II,III\},\,j\in\{Red,Yellow,Blue\}$  

4. **Blending upper‑bound proportions**  
   $\displaystyle x_{ij}\;\leq\;u_{ij}\,\sum_{k\in\{I,II,III\}}x_{kj},\quad \forall i\in\{I,II,III\},\,j\in\{Red,Yellow,Blue\}$  

5. **Non‑negativity**  
   $x_{ij}\geq0,\quad \forall i,j$  

##### Parameters  

- **Grades**: $I,\;II,\;III$  
- **Brands**: $Red,\;Yellow,\;Blue$  

- **Supply limits**  
  $d=\begin{bmatrix}1500\\2000\\1000\end{bmatrix}$  

- **Raw material costs**  
  $c=\begin{bmatrix}6\\4.5\\3\end{bmatrix}$  

- **Selling prices**  
  $p=\begin{bmatrix}5.5\\5\\4.8\end{bmatrix}$  

- **Lower‑bound proportion matrix** $L=[l_{ij}]$  
  $L=\begin{bmatrix}
      0   & 0.20 & 0   \\
      0.50& 0    & 0   \\
      0   & 0    & 0
     \end{bmatrix}$  

- **Upper‑bound proportion matrix** $U=[u_{ij}]$  
  $U=\begin{bmatrix}
      0.10& 0    & 0.50\\
      0   & 0    & 0.10\\
      0   & 0.70 & 0
     \end{bmatrix}$  

(Rows correspond to grades $I,II,III$; columns correspond to brands $Red,Yellow,Blue$.)