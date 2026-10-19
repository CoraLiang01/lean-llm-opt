##### Decision Variables

$y_{ij}\in\{0,1\}$: binary variable equal to 1 if the van travels directly from node $i$ to node $j$, 0 otherwise, for all $i,j\in N=\{\text{Depot},A,B,C\}$ with $i\neq j$.

$u_{k}\in[1,3]$: continuous variable used for subtour elimination, for all customer nodes $k\in\{A,B,C\}$.

##### Objective Function

$\displaystyle \min\sum_{i\in N}\sum_{\substack{j\in N\\j\neq i}}d_{ij}\,y_{ij}$

where the distance parameters $d_{ij}$ are taken from the distance matrix:

\[
\begin{aligned}
d_{\text{Depot},A}&=28,& d_{\text{Depot},B}&=41,& d_{\text{Depot},C}&=63,\\
d_{A,\text{Depot}}&=28,& d_{A,B}&=27,& d_{A,C}&=87,\\
d_{B,\text{Depot}}&=41,& d_{B,A}&=27,& d_{B,C}&=81,\\
d_{C,\text{Depot}}&=63,& d_{C,A}&=87,& d_{C,B}&=81.
\end{aligned}
\]

##### Constraints

1. **Out‑degree constraints (each node has exactly one outgoing arc):**  
   \[
   \sum_{\substack{j\in N\\j\neq i}}y_{ij}=1,\qquad \forall i\in N.
   \]

2. **In‑degree constraints (each node has exactly one incoming arc):**  
   \[
   \sum_{\substack{i\in N\\i\neq j}}y_{ij}=1,\qquad \forall j\in N.
   \]

3. **Subtour elimination (Miller–Tucker–Zemlin constraints):**  
   \[
   u_{i}-u_{j}+3\,y_{ij}\le 2,\qquad \forall i,j\in\{A,B,C\},\; i\neq j.
   \]

4. **Bounds on $u$ variables:**  
   \[
   1\le u_{k}\le 3,\qquad \forall k\in\{A,B,C\}.
   \]

5. **Binary and continuous variable domains:**  
   \[
   y_{ij}\in\{0,1\},\qquad \forall i,j\in N,\; i\neq j;\qquad
   u_{k}\in\mathbb{R},\qquad \forall k\in\{A,B,C\}.
   \]

##### Retrieved Information

```json
{
  "nodes": ["Depot","A","B","C"],
  "distances": {
    "Depot": {"A":28,"B":41,"C":63},
    "A": {"Depot":28,"B":27,"C":87},
    "B": {"Depot":41,"A":27,"C":81},
    "C": {"Depot":63,"A":87,"B":81}
  }
}
```

The model above fully specifies the traveling salesman problem for the courier van visiting customers A, B, and C exactly once and returning to the depot, with the objective of minimizing total travel distance.