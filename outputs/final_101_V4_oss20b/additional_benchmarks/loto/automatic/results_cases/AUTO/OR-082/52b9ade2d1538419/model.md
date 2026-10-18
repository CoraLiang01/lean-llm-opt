##### Decision Variables

$x_{ij}\in\{0,1\}$: 1 if the van travels directly from node $i$ to node $j$, 0 otherwise, for all $i,j\in N,\;i\neq j$.

$u_i\in[1,3]$: continuous “position” variable for customer $i\in V$ used in the MTZ subtour‑elimination constraints.

##### Objective Function

$\displaystyle \min\sum_{i\in N}\sum_{\substack{j\in N\\j\neq i}}d_{ij}\,x_{ij}$

##### Constraints

1. **Depot outflow**  
   $\displaystyle \sum_{j\in V}x_{\text{Depot},j}=1$

2. **Depot inflow**  
   $\displaystyle \sum_{i\in V}x_{i,\text{Depot}}=1$

3. **Customer outflow**  
   $\displaystyle \sum_{\substack{j\in N\\j\neq i}}x_{i,j}=1,\quad\forall i\in V$

4. **Customer inflow**  
   $\displaystyle \sum_{\substack{j\in N\\j\neq i}}x_{j,i}=1,\quad\forall i\in V$

5. **MTZ subtour elimination**  
   $\displaystyle u_i-u_j+3\,x_{ij}\le 2,\quad\forall i,j\in V,\;i\neq j$

6. **Position bounds**  
   $\displaystyle 1\le u_i\le 3,\quad\forall i\in V$

7. **Binary and continuous domains**  
   $x_{ij}\in\{0,1\},\;u_i\in[1,3]$

$N=\{\text{Depot},A,B,C\}$, $V=\{A,B,C\}$.

##### Retrieved Information

```json
{
  "nodes": ["Depot","A","B","C"],
  "customers": ["A","B","C"],
  "distances": {
    "Depot": {"A":28,"B":41,"C":63},
    "A": {"Depot":28,"B":27,"C":87},
    "B": {"Depot":41,"A":27,"C":81},
    "C": {"Depot":63,"A":87,"B":81}
  }
}
```