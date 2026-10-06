Let there be three managers: MA, MB, and MC, and three projects: P1, P2, and P3. Let the cost matrix C be:

\[
C = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 \\
\end{bmatrix}
\]

where \(C_{ij}\) is the cost for manager \(i\) to complete project \(j\), with the following mapping:
- \(i = 1\) for MA, \(i = 2\) for MB, \(i = 3\) for MC
- \(j = 1\) for P1, \(j = 2\) for P2, \(j = 3\) for P3

Define binary decision variables:
\[
x_{ij} = 
\begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]

The mathematical model is:

\[
\text{Minimize} \quad Z = 3000x_{11} + 3200x_{12} + 3100x_{13} + 2800x_{21} + 3300x_{22} + 2900x_{23} + 2900x_{31} + 3100x_{32} + 3000x_{33}
\]

Subject to:
\[
x_{11} + x_{12} + x_{13} = 1 \quad \text{(MA assigned to exactly one project)}
\]
\[
x_{21} + x_{22} + x_{23} = 1 \quad \text{(MB assigned to exactly one project)}
\]
\[
x_{31} + x_{32} + x_{33} = 1 \quad \text{(MC assigned to exactly one project)}
\]
\[
x_{11} + x_{21} + x_{31} = 1 \quad \text{(P1 assigned to exactly one manager)}
\]
\[
x_{12} + x_{22} + x_{32} = 1 \quad \text{(P2 assigned to exactly one manager)}
\]
\[
x_{13} + x_{23} + x_{33} = 1 \quad \text{(P3 assigned to exactly one manager)}
\]
\[
x_{ij} \in \{0,1\} \quad \forall i \in \{1,2,3\},\ j \in \{1,2,3\}
\]

Where:
- \(x_{ij}\) are binary variables indicating assignment,
- The cost matrix is:
  \[
  \begin{bmatrix}
  3000 & 3200 & 3100 \\
  2800 & 3300 & 2900 \\
  2900 & 3100 & 3000 \\
  \end{bmatrix}
  \]
- The objective is to minimize the total cost of assignments while ensuring each manager is assigned to exactly one project and each project is assigned to exactly one manager.