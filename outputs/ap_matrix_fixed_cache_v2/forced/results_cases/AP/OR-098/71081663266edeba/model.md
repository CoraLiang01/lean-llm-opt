##### Variables

Let $w_j$ denote the daily wage (yuan) of worker $j$, for all workers $j$ in the set $W$ (ordered as in the columns of the CSV). Let $W = [\text{Carpenter}, \text{Electrician}, \text{Painter}, \text{Worker\_004}, \ldots, \text{Worker\_150}]$.

##### Parameters

Let $d_{ij}$ denote the number of days worker $j$ worked on homeowner $i$'s home, for all $i, j \in W$. The matrix $D = [d_{ij}]$ is given by the CSV, with rows and columns in the same order.

The daily wage of the first worker (Carpenter) is fixed: $w_1 = 60.00$.

Each worker's total days worked is exactly 10:
$\sum_{i \in W} d_{ij} = 10$ for all $j \in W$.

##### Model

###### 1. Wage scale constraint

$w_1 = 60.00$

###### 2. Mutual payment balance constraints

For every participant $k \in W$ (for $k = 1, \ldots, 150$):

$\sum_{\substack{i=1 \\ i \neq k}}^{n} d_{ik} w_k = \sum_{j=1}^{n} d_{kj} w_j - d_{kk} w_k$

That is, for each $k$:
- The left side is the total income worker $k$ receives for working on others' homes (sum over all $i \neq k$ of days $d_{ik}$ times their own wage $w_k$).
- The right side is the total amount worker $k$ pays for work done at their own home (sum over all $j$ of days $d_{kj}$ times $w_j$), minus the self-payment $d_{kk} w_k$ (since self-payment is not an actual transfer).

Equivalently, for all $k$:
$\sum_{i=1}^{n} d_{ik} w_k = \sum_{j=1}^{n} d_{kj} w_j$

###### 3. Non-negativity

$w_j \geq 0$ for all $j \in W$

##### Complete Model

Given:
- $W = [\text{Carpenter}, \text{Electrician}, \text{Painter}, \text{Worker\_004}, \ldots, \text{Worker\_150}]$
- $D = [d_{ij}]$ as retrieved below

Find $w_j$ for all $j \in W$ such that:

1. $w_1 = 60.00$
2. For all $k = 1, \ldots, 150$:
   $$
   \sum_{i=1}^{n} d_{ik} w_k = \sum_{j=1}^{n} d_{kj} w_j
   $$
3. $w_j \geq 0$ for all $j$

##### Retrieved Information

- **Workers (ordered):**
  ["Carpenter", "Electrician", "Painter", "Worker_004", "Worker_005", "Worker_006", "Worker_007", "Worker_008", "Worker_009", "Worker_010", "Worker_011", "Worker_012", "Worker_013", "Worker_014", "Worker_015", "Worker_016", "Worker_017", "Worker_018", "Worker_019", "Worker_020", "Worker_021", "Worker_022", "Worker_023", "Worker_024", "Worker_025", "Worker_026", "Worker_027", "Worker_028", "Worker_029", "Worker_030", "Worker_031", "Worker_032", "Worker_033", "Worker_034", "Worker_035", "Worker_036", "Worker_037", "Worker_038", "Worker_039", "Worker_040", "Worker_041", "Worker_042", "Worker_043", "Worker_044", "Worker_045", "Worker_046", "Worker_047", "Worker_048", "Worker_049", "Worker_050", "Worker_051", "Worker_052", "Worker_053", "Worker_054", "Worker_055", "Worker_056", "Worker_057", "Worker_058", "Worker_059", "Worker_060", "Worker_061", "Worker_062", "Worker_063", "Worker_064", "Worker_065", "Worker_066", "Worker_067", "Worker_068", "Worker_069", "Worker_070", "Worker_071", "Worker_072", "Worker_073", "Worker_074", "Worker_075", "Worker_076", "Worker_077", "Worker_078", "Worker_079", "Worker_080", "Worker_081", "Worker_082", "Worker_083", "Worker_084", "Worker_085", "Worker_086", "Worker_087", "Worker_088", "Worker_089", "Worker_090", "Worker_091", "Worker_092", "Worker_093", "Worker_094", "Worker_095", "Worker_096", "Worker_097", "Worker_098", "Worker_099", "Worker_100", "Worker_101", "Worker_102", "Worker_103", "Worker_104", "Worker_105", "Worker_106", "Worker_107", "Worker_108", "Worker_109", "Worker_110", "Worker_111", "Worker_112", "Worker_113", "Worker_114", "Worker_115", "Worker_116", "Worker_117", "Worker_118", "Worker_119", "Worker_120", "Worker_121", "Worker_122", "Worker_123", "Worker_124", "Worker_125", "Worker_126", "Worker_127", "Worker_128", "Worker_129", "Worker_130", "Worker_131", "Worker_132", "Worker_133", "Worker_134", "Worker_135", "Worker_136", "Worker_137", "Worker_138", "Worker_139", "Worker_140", "Worker_141", "Worker_142", "Worker_143", "Worker_144", "Worker_145", "Worker_146", "Worker_147", "Worker_148", "Worker_149", "Worker_150"]

- **Work days matrix $D = [d_{ij}]$**:  
  Each row corresponds to a homeowner (in the same order as above), each column to a worker (same order), and each entry $d_{ij}$ is the number of days worker $j$ worked on homeowner $i$'s home.  
  (See the retrieved CSV data above for all values.)

- **Total days per worker:**  
  For each $j$, $\sum_{i} d_{ij} = 10$.

##### Summary

- Variables: $w_j$ (daily wage for each worker $j$), with $w_1 = 60.00$.
- For each participant $k$, enforce: $\sum_{i} d_{ik} w_k = \sum_{j} d_{kj} w_j$.
- All $w_j \geq 0$.
- All $d_{ij}$ as given in the CSV.

This is a system of $n$ linear equations in $n$ variables (with $w_1$ fixed), fully specified by the data above.