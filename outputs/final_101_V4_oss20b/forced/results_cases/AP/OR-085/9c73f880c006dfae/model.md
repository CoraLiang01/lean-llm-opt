**Mathematical Model**

Let  

* \(N=\{1,2,\dots ,15\}\) – set of locations (customer 1 is the depot).  
* \(M=N\setminus\{1\}\) – set of customer nodes.

**Parameters**

\[
d_{ij}\quad\forall i,j\in N,\; i\neq j
\]

*Distance between location \(i\) and \(j\) (symmetric).*

**Decision Variables**

\[
x_{ij}\in\{0,1\}\quad\forall i,j\in N,\; i\neq j
\]

*1 if the tour goes directly from \(i\) to \(j\), 0 otherwise.*

\[
u_i\in[2,\,|N|-1]\quad\forall i\in M
\]

*MTZ order variables for subtour elimination.*

---

### Objective

\[
\min \;\sum_{i\in N}\;\sum_{\substack{j\in N\\ j\neq i}} d_{ij}\,x_{ij}
\]

---

### Constraints

1. **Out‑degree (each node has one outgoing edge)**  

   \[
   \sum_{\substack{j\in N\\ j\neq i}} x_{ij}=1 \qquad \forall i\in N
   \]

2. **In‑degree (each node has one incoming edge)**  

   \[
   \sum_{\substack{i\in N\\ i\neq j}} x_{ij}=1 \qquad \forall j\in N
   \]

3. **MTZ subtour elimination**  

   \[
   u_i - u_j + (|N|-1)\,x_{ij}\;\le\;|N|-2
   \qquad \forall i,j\in M,\; i\neq j
   \]

4. **Bounds on order variables**  

   \[
   2 \;\le\; u_i \;\le\; |N|-1 \qquad \forall i\in M
   \]

5. **Binary variables**  

   \[
   x_{ij}\in\{0,1\}\qquad \forall i,j\in N,\; i\neq j
   \]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(d_{ij}\) | Table **file_0_view_0** (20.csv) | For each pair \((i,j)\) with \(i<j\): <br>• Row index = \(i-1\) (0‑based) <br>• Column name = string of \(j\) <br>• Value = cell content (numeric). <br>For \(i>j\), set \(d_{ij}=d_{ji}\). |
| \(N\) | Implicit from file | \(N=\{1,\dots,15\}\) |
| \(M\) | Implicit | \(M=N\setminus\{1\}\) |

*All other parameters are derived from the above mapping; no additional numerical tables are enumerated.*