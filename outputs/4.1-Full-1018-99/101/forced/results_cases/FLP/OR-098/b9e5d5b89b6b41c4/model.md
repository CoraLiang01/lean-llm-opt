##### Parameters

Let $N$ be the number of participants (homeowners/workers), indexed by $i=1,\ldots,N$.
Let $W = \{$Carpenter, Electrician, Painter, Worker_004, \ldots, Worker_{150}$\}$ be the ordered list of workers (columns in the CSV).
Let $O = \{$Carpenter, Electrician, Painter, Worker_004, \ldots, Worker_{150}$\}$ be the ordered list of homeowners (rows in the CSV).
Let $a_{ij}$ be the number of days worker $j$ worked on homeowner $i$'s home, as given in the CSV file (row $i$, column $j$).
Let $w_j$ be the daily wage of worker $j$ (decision variable).

The daily wage of the first worker (Carpenter) is fixed: $w_1 = 60.00$.

Each worker contributes exactly 10 days in total:
\[
\sum_{i=1}^N a_{ij} = 10, \quad \forall j=1,\ldots,N
\]

##### Decision Variables

$w_j \in \mathbb{R}$: daily wage of worker $j$, for $j=1,\ldots,N$.

##### Constraints

1. **Wage scale anchor:**
   \[
   w_1 = 60.00
   \]
   (where $w_1$ is the wage of the first worker, e.g., Carpenter)

2. **Mutual payment balance for each participant $k=1,\ldots,N$:**
   \[
   \sum_{\substack{i=1 \\ i \neq k}}^N a_{ik} w_k = \sum_{j=1}^N a_{kj} w_j, \quad \forall k=1,\ldots,N
   \]
   That is, for each participant $k$:
   - The left side is the total income worker $k$ earns from working on others' homes (sum over all $i \neq k$ of days $a_{ik}$ times $w_k$).
   - The right side is the total expenditure participant $k$ pays for work performed at their own home (sum over all workers $j$ of days $a_{kj}$ times $w_j$).

   Equivalently, since $a_{kk}$ is the days worker $k$ worked on their own home:
   \[
   \sum_{i=1}^N a_{ik} w_k - a_{kk} w_k = \sum_{j=1}^N a_{kj} w_j
   \]
   \[
   \left(\sum_{i=1}^N a_{ik} - a_{kk}\right) w_k = \sum_{j=1}^N a_{kj} w_j
   \]
   But since $\sum_{i=1}^N a_{ik} = 10$ for all $k$, this is:
   \[
   (10 - a_{kk}) w_k = \sum_{j=1}^N a_{kj} w_j
   \]
   for all $k=1,\ldots,N$.

##### Complete Model

Given the data $a_{ij}$ from the CSV, the model is:

- Variables: $w_j$ for $j=1,\ldots,N$
- Parameters: $a_{ij}$ for $i,j=1,\ldots,N$ (from the CSV)
- Anchor: $w_1 = 60.00$
- For all $k=1,\ldots,N$:
  \[
  (10 - a_{kk}) w_k = \sum_{j=1}^N a_{kj} w_j
  \]

##### Retrieved Information

- $N = 150$
- $W = $ [Carpenter, Electrician, Painter, Worker_004, ..., Worker_150] (columns in CSV, in order)
- $O = $ [Carpenter, Electrician, Painter, Worker_004, ..., Worker_150] (rows in CSV, in order)
- $a_{ij}$: see work_days.csv, where $a_{ij}$ is the entry in row $i$ (Owner), column $j$ (Worker)
- For all $j$, $\sum_{i=1}^N a_{ij} = 10$
- $w_1 = 60.00$

##### Matrix Form

Let $A$ be the $N \times N$ matrix with entries $a_{ij}$.

For $k=1,\ldots,N$:
\[
(10 - a_{kk}) w_k - \sum_{j=1}^N a_{kj} w_j = 0
\]
with $w_1 = 60.00$.

##### Summary

Given the full $a_{ij}$ matrix from work_days.csv, solve the above linear system for $w_j$ ($j=2,\ldots,N$), with $w_1$ fixed at $60.00$, to obtain the fair daily wage for each worker such that each participant's total income from working on others' homes equals their total expenditure for work performed at their own home.