##### Decision Variables

$x_{ij}\in\{0,1\}$: binary variable that equals 1 if the van travels directly from node $i$ to node $j$, and 0 otherwise, for all $i,j\in N$ with $i\neq j$.

$u_i\in\mathbb{R}$: continuous “position” variable for node $i$ used in the Miller–Tucker–Zemlin subtour elimination, for all $i\in\{A,B,C\}$.

##### Objective Function

$\displaystyle \min \sum_{i\in N}\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}$

##### Constraints

1. **Outgoing flow**: $\displaystyle \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1,\quad \forall i\in N$.

2. **Incoming flow**: $\displaystyle \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1,\quad \forall j\in N$.

3. **MTZ subtour elimination**:  
   $\displaystyle u_i - u_j + (|N|-1)\,x_{ij}\le |N|-2,\quad \forall i,j\in\{A,B,C\},\, i\neq j$.

4. **Position bounds**: $\displaystyle 1\le u_i\le |N|-1,\quad \forall i\in\{A,B,C\}$.

5. **Domain**: $x_{ij}\in\{0,1\}$ for all $i\neq j$, $u_i\in\mathbb{R}$ for all $i\in\{A,B,C\}$.

$N=\{\text{Depot},A,B,C\}$, $|N|=4$.

##### Retrieved Information

Distance matrix (rows and columns correspond to nodes in the order Depot, A, B, C):

\[
\begin{array}{c|cccc}
 & \text{Depot} & A & B & C\\ \hline
\text{Depot} & 0 & 28 & 41 & 63\\
A & 28 & 0 & 27 & 87\\
B & 41 & 27 & 0 & 81\\
C & 63 & 87 & 81 & 0
\end{array}
\]