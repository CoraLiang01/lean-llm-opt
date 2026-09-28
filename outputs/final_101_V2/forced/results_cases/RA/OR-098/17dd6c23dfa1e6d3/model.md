Let $N$ be the number of participants (workers/homeowners), with identifiers and order as in the columns of work_days.csv: $w_1, w_2, \ldots, w_N$. Let $d_{ij}$ denote the number of days worker $w_j$ spent on homeowner $w_i$'s home, as given by the entry in row $i$, column $j$ of work_days.csv. Let $x_j$ denote the daily wage of worker $w_j$ (yuan/day).

The first worker listed ($w_1$, e.g., "Carpenter") has wage fixed at $x_1 = 60.00$.

The model is:

#### Variables

- $x_j \in \mathbb{R}$, for $j = 1, \ldots, N$ (daily wage of worker $w_j$)
- $x_1 = 60.00$

#### Constraints

For each participant $i = 1, \ldots, N$:

\[
\sum_{\substack{j=1 \\ j \neq i}}^{N} d_{ji} \, x_i = \sum_{\substack{j=1 \\ j \neq i}}^{N} d_{ij} \, x_j
\]

That is, for each $i$:
- The left side is the total income participant $i$ receives for working on others' homes (sum over all $j \neq i$ of days $i$ worked on $j$'s home, times $i$'s wage).
- The right side is the total payment participant $i$ makes to others for work performed at their own home (sum over all $j \neq i$ of days $j$ worked on $i$'s home, times $j$'s wage).

#### Fixed wage

\[
x_1 = 60.00
\]

#### Data

- $d_{ij}$: as given in work_days.csv, with $i$ indexing rows ("Owner") and $j$ indexing columns (worker names), preserving the original order and identifiers.

#### Complete Model

\[
\begin{align*}
&\text{Find } x_j \in \mathbb{R}, \quad j = 1, \ldots, N \\
&\text{such that:} \\
&\quad \sum_{\substack{j=1 \\ j \neq i}}^{N} d_{ji} \, x_i = \sum_{\substack{j=1 \\ j \neq i}}^{N} d_{ij} \, x_j, \quad \forall i = 1, \ldots, N \\
&\quad x_1 = 60.00 \\
&\text{where } d_{ij} = \text{entry in row } i, \text{ column } j \text{ of work_days.csv}
\end{align*}
\]

All data and identifiers are as retrieved from work_days.csv, preserving original row and column order.