Let:
- M = {1, 2, ..., 12} be the set of machines, indexed by i.
- T = {1, 2, ..., 12} be the set of tasks, indexed by j.
- Let the mapping from tasks to indices be: A=1, B=2, ..., L=12.
- Let c_ij be the cost of assigning machine i to task j, as given below.
- Let x_ij ∈ {0,1} be a binary variable: x_ij = 1 if machine i is assigned to task j, 0 otherwise.

Cost matrix C = [c_ij] (rows: machines i=1..12, columns: tasks j=1..12):

\[
C = \begin{bmatrix}
167.4 & 98.6 & 189.4 & 119.6 & 182.0 & 145.1 & 185.4 & 94.8 & 122.3 & 123.3 & 96.1 & 90.3 \\
156.2 & 88.7 & 187.3 & 124.7 & 173.2 & 144.3 & 179.0 & 91.5 & 115.1 & 119.5 & 100.1 & 88.6 \\
184.3 & 121.0 & 216.6 & 140.0 & 196.2 & 168.8 & 205.6 & 114.2 & 133.3 & 144.5 & 116.0 & 107.7 \\
157.9 & 92.9 & 185.1 & 120.3 & 175.1 & 146.2 & 180.8 & 86.3 & 111.6 & 115.9 & 98.1 & 91.1 \\
175.6 & 103.6 & 204.5 & 130.0 & 192.8 & 157.5 & 194.2 & 106.9 & 129.9 & 134.9 & 105.8 & 98.6 \\
166.8 & 107.0 & 199.2 & 130.4 & 183.6 & 159.5 & 187.0 & 98.2 & 121.3 & 126.2 & 105.9 & 101.8 \\
159.7 & 93.2 & 183.8 & 113.0 & 171.9 & 139.1 & 169.6 & 85.1 & 110.0 & 116.7 & 90.6 & 85.2 \\
184.8 & 115.9 & 205.1 & 138.6 & 195.4 & 160.1 & 200.2 & 108.5 & 136.9 & 140.0 & 114.6 & 103.9 \\
157.3 & 86.2 & 186.0 & 113.9 & 166.2 & 136.8 & 167.5 & 78.8 & 107.4 & 114.5 & 87.2 & 78.6 \\
164.8 & 97.8 & 200.9 & 125.8 & 188.9 & 151.2 & 187.7 & 99.5 & 119.5 & 132.1 & 101.1 & 98.4 \\
164.0 & 92.2 & 186.2 & 115.7 & 174.5 & 143.0 & 175.9 & 92.3 & 114.0 & 121.2 & 93.7 & 91.2 \\
151.7 & 76.7 & 179.5 & 109.5 & 160.6 & 128.4 & 170.2 & 74.4 & 103.7 & 110.4 & 83.7 & 75.2 \\
\end{bmatrix}
\]

Mathematical Model:

Variables:
- For all i ∈ M, j ∈ T: x_ij ∈ {0,1}

Objective:
\[
\min \sum_{i=1}^{12} \sum_{j=1}^{12} c_{ij} x_{ij}
\]

Subject to:
1. Each machine is assigned to exactly one task:
\[
\forall i \in M: \quad \sum_{j=1}^{12} x_{ij} = 1
\]
2. Each task is assigned to exactly one machine:
\[
\forall j \in T: \quad \sum_{i=1}^{12} x_{ij} = 1
\]
3. Binary assignment variables:
\[
x_{ij} \in \{0,1\} \quad \forall i \in M,\, j \in T
\]

Where:
- c_ij is as given in the cost matrix above (row i, column j).
- x_ij = 1 if machine i is assigned to task j, 0 otherwise.

This is a classical linear assignment problem (minimum-cost bipartite matching) with a 12x12 cost matrix.