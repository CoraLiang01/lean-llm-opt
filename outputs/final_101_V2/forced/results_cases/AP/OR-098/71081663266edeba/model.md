##### Variables

Let $n$ be the total number of participants (workers/homeowners), indexed by $i = 1, 2, \ldots, n$.

Let $W = \{$"Carpenter", "Electrician", "Painter", "Worker_004", ..., "Worker_150"$\}$ be the ordered set of all workers, as given by the columns of the CSV.

Let $d_{ij}$ denote the number of days worker $j$ worked on homeowner $i$'s home, as given by the entry in row $i$, column $j$ of the CSV.

Let $w_j$ denote the daily wage (in yuan) of worker $j$.

##### Parameters

- $d_{ij}$: Number of days worker $j$ worked on homeowner $i$'s home, for all $i, j \in W$.
- $w_1 = 60.00$: The daily wage of the first worker in the file (e.g., "Carpenter") is fixed at 60.00 yuan.

##### Objective

No explicit objective function is required; the problem is to find a feasible solution for the wage vector $\mathbf{w}$ that satisfies all constraints.

##### Constraints

###### 1. Wage normalization

$w_1 = 60.00$

###### 2. Mutual payment balance for each participant

For each participant $i \in W$:

$\sum_{\substack{j=1 \\ j \ne i}}^{n} d_{ji} w_i = \sum_{\substack{j=1 \\ j \ne i}}^{n} d_{ij} w_j$

That is, for each $i$:
- The total income of worker $i$ from working on others' homes (sum over all $j \ne i$ of days $d_{ji}$ worked by $i$ at $j$'s home, times $w_i$) equals
- The total expenditure of $i$ for work performed at their own home (sum over all $j \ne i$ of days $d_{ij}$ worked by $j$ at $i$'s home, times $w_j$).

###### 3. Total work days per worker

For each worker $j \in W$:

$\sum_{i=1}^{n} d_{ij} = 10$

###### 4. Non-negativity

$w_j \geq 0 \quad \forall j \in W$

##### Retrieved Information

```json
{
  "workers": [
    "Carpenter", "Electrician", "Painter", "Worker_004", "Worker_005", "Worker_006", "Worker_007", "Worker_008", "Worker_009", "Worker_010", "Worker_011", "Worker_012", "Worker_013", "Worker_014", "Worker_015", "Worker_016", "Worker_017", "Worker_018", "Worker_019", "Worker_020", "Worker_021", "Worker_022", "Worker_023", "Worker_024", "Worker_025", "Worker_026", "Worker_027", "Worker_028", "Worker_029", "Worker_030", "Worker_031", "Worker_032", "Worker_033", "Worker_034", "Worker_035", "Worker_036", "Worker_037", "Worker_038", "Worker_039", "Worker_040", "Worker_041", "Worker_042", "Worker_043", "Worker_044", "Worker_045", "Worker_046", "Worker_047", "Worker_048", "Worker_049", "Worker_050", "Worker_051", "Worker_052", "Worker_053", "Worker_054", "Worker_055", "Worker_056", "Worker_057", "Worker_058", "Worker_059", "Worker_060", "Worker_061", "Worker_062", "Worker_063", "Worker_064", "Worker_065", "Worker_066", "Worker_067", "Worker_068", "Worker_069", "Worker_070", "Worker_071", "Worker_072", "Worker_073", "Worker_074", "Worker_075", "Worker_076", "Worker_077", "Worker_078", "Worker_079", "Worker_080", "Worker_081", "Worker_082", "Worker_083", "Worker_084", "Worker_085", "Worker_086", "Worker_087", "Worker_088", "Worker_089", "Worker_090", "Worker_091", "Worker_092", "Worker_093", "Worker_094", "Worker_095", "Worker_096", "Worker_097", "Worker_098", "Worker_099", "Worker_100", "Worker_101", "Worker_102", "Worker_103", "Worker_104", "Worker_105", "Worker_106", "Worker_107", "Worker_108", "Worker_109", "Worker_110", "Worker_111", "Worker_112", "Worker_113", "Worker_114", "Worker_115", "Worker_116", "Worker_117", "Worker_118", "Worker_119", "Worker_120", "Worker_121", "Worker_122", "Worker_123", "Worker_124", "Worker_125", "Worker_126", "Worker_127", "Worker_128", "Worker_129", "Worker_130", "Worker_131", "Worker_132", "Worker_133", "Worker_134", "Worker_135", "Worker_136", "Worker_137", "Worker_138", "Worker_139", "Worker_140", "Worker_141", "Worker_142", "Worker_143", "Worker_144", "Worker_145", "Worker_146", "Worker_147", "Worker_148", "Worker_149", "Worker_150"
  ],
  "days_matrix": {
    "Carpenter": {"Carpenter": 1, "Electrician": 0, "Painter": 0, "Worker_004": 0, ..., "Worker_150": 1},
    "Electrician": {"Carpenter": 1, "Electrician": 1, "Painter": 1, "Worker_004": 0, ..., "Worker_150": 0},
    "Painter": {"Carpenter": 0, "Electrician": 1, "Painter": 1, "Worker_004": 0, ..., "Worker_150": 0},
    // ... (all other rows, up to Worker_150)
  }
}
```

##### Full Mathematical Model

Let $n$ be the number of participants, $W = \{1,2,\ldots,n\}$, and let the mapping between indices and worker names be as in the CSV order.

Variables:
- $w_j \geq 0$ for $j \in W$ (daily wage of worker $j$)

Parameters:
- $d_{ij}$: days worker $j$ worked on homeowner $i$'s home (from CSV)
- $w_1 = 60.00$

Constraints:
- $w_1 = 60.00$
- For all $i \in W$:
  $$
  \sum_{\substack{j=1 \\ j \ne i}}^{n} d_{ji} w_i = \sum_{\substack{j=1 \\ j \ne i}}^{n} d_{ij} w_j
  $$
- For all $j \in W$:
  $$
  \sum_{i=1}^{n} d_{ij} = 10
  $$
- $w_j \geq 0$ for all $j \in W$

All identifiers and values are preserved as in the original data.