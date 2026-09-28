Let $N$ be the number of participants (workers/homeowners), indexed in the order of columns in work_days.csv: $W_1, W_2, \ldots, W_N$. Let $d_{ij}$ be the number of days worker $j$ worked on homeowner $i$'s home (from row $i$, column $j$). Let $w_j$ be the daily wage of worker $j$.

Given: $w_1 = 60.00$.

Each worker contributes exactly 10 days: $\sum_{i=1}^N d_{ij} = 10$ for all $j=1,\ldots,N$.

For each participant $k=1,\ldots,N$, their total income from working on others’ homes equals their total expenditure for work performed at their own home:
\[
\sum_{\substack{i=1 \\ i\neq k}}^N d_{ik} w_i = \sum_{j=1}^N d_{kj} w_j
\]
or, equivalently,
\[
\sum_{i=1}^N d_{ik} w_i - d_{kk} w_k = \sum_{j=1}^N d_{kj} w_j
\]
\[
\sum_{i=1}^N d_{ik} w_i - \sum_{j=1}^N d_{kj} w_j = d_{kk} w_k
\]

But the standard form is:
For each $k=1,\ldots,N$,
\[
\sum_{i=1}^N d_{ik} w_i = \sum_{j=1}^N d_{kj} w_j
\]

With $w_1 = 60.00$.

All $w_j$ are real variables.

#### Model

Variables:
- $w_j$ (real), daily wage of worker $j$, for $j=1,\ldots,N$.

Parameters:
- $d_{ij}$: number of days worker $j$ worked on homeowner $i$'s home (from work_days.csv).

Constraints:
- Wage normalization: $w_1 = 60.00$
- For each $k=1,\ldots,N$:
  \[
  \sum_{i=1}^N d_{ik} w_i = \sum_{j=1}^N d_{kj} w_j
  \]

Data (first three rows, all columns shown for illustration):

| Owner      | Carpenter | Electrician | Painter | Worker_004 | ... | Worker_150 |
|------------|-----------|-------------|---------|------------|-----|------------|
| Carpenter  | 1         | 0           | 0       | 0          | ... | 1          |
| Electrician| 1         | 1           | 1       | 0          | ... | 0          |
| Painter    | 0         | 1           | 1       | 0          | ... | 0          |

Let the column order be $W_1=$Carpenter, $W_2=$Electrician, $W_3=$Painter, $W_4=$Worker_004, ..., $W_{150}=$Worker_150.

Full model:

Given $d_{ij}$ from work_days.csv, find $w_j$ for $j=1,\ldots,N$ such that:
- $w_1 = 60.00$
- For all $k=1,\ldots,N$:
  \[
  \sum_{i=1}^N d_{ik} w_i = \sum_{j=1}^N d_{kj} w_j
  \]

where $w_j$ are real variables.

All identifiers and coefficients are as in the retrieved data.