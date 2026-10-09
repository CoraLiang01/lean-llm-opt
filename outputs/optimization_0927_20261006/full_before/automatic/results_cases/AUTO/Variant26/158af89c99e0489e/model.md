##### Sets

Let  
- $W$ = set of eligible workers (on_leave=0):  
$\{ \text{W00}, \text{W04}, \text{W06}, \text{W10}, \text{W11}, \text{W14}, \text{W15}, \text{W19}, \text{W20} \}$

- $P$ = set of projects:  
$\{ \text{P00}, \text{P01}, \text{P02}, \text{P03}, \text{P04}, \text{P05}, \text{P06}, \text{P07} \}$

- $O$ = set of valid offers (worker-project pairs listed below), where worker_skill $\geq$ required_skill and on_leave=0.

##### Parameters

For each $(w,p) \in O$:

- $c_{w,p}$ = cost in USD cents for worker $w$ to do project $p$ (see table below).

##### Decision Variables

For each $(w,p) \in O$:

- $x_{w,p} \in \{0,1\}$, where $x_{w,p}=1$ if worker $w$ is assigned to project $p$, $0$ otherwise.

##### Objective Function

$\min \sum_{(w,p) \in O} c_{w,p} \cdot x_{w,p}$

##### Constraints

1. **Each project assigned to exactly one worker (from valid offers):**

$\sum_{\substack{w: (w,p) \in O}} x_{w,p} = 1 \quad \forall p \in P$

2. **Each worker assigned to at most one project:**

$\sum_{\substack{p: (w,p) \in O}} x_{w,p} \leq 1 \quad \forall w \in W$

3. **Assignment only allowed for valid offers:**

$x_{w,p} = 0$ for all $(w,p) \notin O$

4. **Variable domain:**

$x_{w,p} \in \{0,1\} \quad \forall (w,p) \in O$

##### Retrieved Information

**Eligible Workers (on_leave=0):**
- W00 (Junior)
- W04 (Junior)
- W06 (Expert)
- W10 (Expert)
- W11 (Senior)
- W14 (Intermediate)
- W15 (Expert)
- W19 (Junior)
- W20 (Intermediate)

**Projects and Required Skills:**
- P00: Intermediate
- P01: Intermediate
- P02: Senior
- P03: Junior
- P04: Junior
- P05: Junior
- P06: Junior
- P07: Junior

**Valid Offers (worker_skill $\geq$ required_skill, on_leave=0):**

| worker_id | worker_skill | project_id | required_skill | cost_cents |
|-----------|-------------|------------|---------------|------------|
| W00       | Junior      | P04        | Junior        | 430        |
| W00       | Junior      | P05        | Junior        | 1385       |
| W00       | Junior      | P06        | Junior        | 260        |
| W00       | Junior      | P07        | Junior        | 1107       |
| W04       | Junior      | P03        | Junior        | 543        |
| W04       | Junior      | P04        | Junior        | 205        |
| W04       | Junior      | P05        | Junior        | 1149       |
| W04       | Junior      | P06        | Junior        | 533        |
| W04       | Junior      | P07        | Junior        | 732        |
| W06       | Expert      | P00        | Intermediate  | 1217       |
| W06       | Expert      | P01        | Intermediate  | 425        |
| W06       | Expert      | P02        | Senior        | 1320       |
| W06       | Expert      | P03        | Junior        | 1097       |
| W06       | Expert      | P04        | Junior        | 822        |
| W06       | Expert      | P05        | Junior        | 221        |
| W06       | Expert      | P06        | Junior        | 496        |
| W06       | Expert      | P07        | Junior        | 908        |
| W10       | Expert      | P00        | Intermediate  | 147        |
| W10       | Expert      | P01        | Intermediate  | 114        |
| W10       | Expert      | P03        | Junior        | 998        |
| W10       | Expert      | P04        | Junior        | 1270       |
| W10       | Expert      | P07        | Junior        | 953        |
| W11       | Senior      | P00        | Intermediate  | 235        |
| W11       | Senior      | P02        | Senior        | 1034       |
| W11       | Senior      | P03        | Junior        | 782        |
| W11       | Senior      | P04        | Junior        | 556        |
| W11       | Senior      | P05        | Junior        | 1018       |
| W11       | Senior      | P07        | Junior        | 1136       |
| W14       | Intermediate| P01        | Intermediate  | 1361       |
| W14       | Intermediate| P03        | Junior        | 379        |
| W14       | Intermediate| P04        | Junior        | 239        |
| W14       | Intermediate| P05        | Junior        | 460        |
| W14       | Intermediate| P06        | Junior        | 130        |
| W15       | Expert      | P00        | Intermediate  | 722        |
| W15       | Expert      | P02        | Senior        | 577        |
| W15       | Expert      | P03        | Junior        | 1062       |
| W15       | Expert      | P04        | Junior        | 1314       |
| W15       | Expert      | P05        | Junior        | 528        |
| W15       | Expert      | P06        | Junior        | 835        |
| W15       | Expert      | P07        | Junior        | 918        |
| W19       | Junior      | P03        | Junior        | 109        |
| W19       | Junior      | P04        | Junior        | 1306       |
| W19       | Junior      | P05        | Junior        | 626        |
| W19       | Junior      | P06        | Junior        | 466        |
| W19       | Junior      | P07        | Junior        | 1365       |
| W20       | Intermediate| P00        | Intermediate  | 675        |
| W20       | Intermediate| P01        | Intermediate  | 1055       |
| W20       | Intermediate| P03        | Junior        | 703        |
| W20       | Intermediate| P06        | Junior        | 893        |
| W20       | Intermediate| P07        | Junior        | 129        |

**Summary of Model:**

- Assign each project to exactly one eligible worker, using only the valid offers above.
- Each worker may be assigned to at most one project.
- Only offers where the worker is not on leave and has sufficient skill are allowed.
- Minimize the total cost in USD cents.

**All parameters, sets, and constraints are explicitly listed above.**