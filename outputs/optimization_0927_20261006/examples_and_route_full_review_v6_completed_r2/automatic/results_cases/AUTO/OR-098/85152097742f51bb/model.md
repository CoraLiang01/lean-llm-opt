Let $w_j$ denote the daily wage (in yuan) for worker $j$, for all workers $j$ listed as columns in the file (in the original order: Carpenter, Electrician, Painter, Worker_004, ..., Worker_150).

Let $d_{ij}$ denote the number of days worker $j$ spent working on homeowner $i$'s home, where $i$ and $j$ both range over all participants (rows and columns of the file, with $i$ given by the Owner column and $j$ by the column headers).

The model is:

**Variables:**
- $w_j \in \mathbb{R}$, for all workers $j$.
- $w_1 = 60.00$ (the daily wage of the first worker, e.g., Carpenter, is fixed at 60.00 yuan).

**Parameters:**
- $d_{ij}$: as given in the CSV, for all $i$ (Owner) and $j$ (worker columns), preserving order and identifiers.

**Equations:**

For every participant $k$ (for each row $k$ in the Owner column):

\[
\sum_{\substack{j=1 \\ j \neq k}}^{N} d_{jk} w_k = \sum_{\substack{j=1 \\ j \neq k}}^{N} d_{kj} w_j
\]
where $N$ is the total number of participants (here, $N=150$), and $d_{jk}$ is the number of days worker $k$ worked on homeowner $j$'s home, $d_{kj}$ is the number of days worker $j$ worked on homeowner $k$'s home.

Equivalently, for each $k$:
\[
\left( \sum_{j \neq k} d_{jk} \right) w_k - \sum_{j \neq k} d_{kj} w_j = 0
\]

**Fixed wage:**
\[
w_1 = 60.00
\]

**(Optional, but implied by the problem statement):**
Each worker contributes exactly 10 work days in total:
\[
\sum_{i=1}^{N} d_{ij} = 10, \quad \forall j=1,\ldots,N
\]
(This is a data property, not a constraint on $w_j$.)

**Domain:**
\[
w_j \in \mathbb{R}, \quad \forall j=1,\ldots,N
\]

**Data:**
All $d_{ij}$ as given in the retrieved CSV, with $i$ as Owner and $j$ as the column headers, in the original order.

**Summary:**
Find $w_j$ for all workers $j$ (with $w_1 = 60.00$) such that, for every participant $k$, the total income they earn from working on others' homes equals the total they pay for work performed at their own home, using the exact work day matrix as given.