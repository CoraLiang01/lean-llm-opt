**Mathematical Model**

```
Sets
  M  : Managers
  P  : Projects

Parameters
  c[m,p]  : Cost of assigning manager m to project p

Decision Variables
  x[m,p]  : 1 if manager m is assigned to project p, 0 otherwise

Objective
  minimize   Σ_{m∈M} Σ_{p∈P}  c[m,p] · x[m,p]

Constraints
  (1)  Σ_{p∈P}  x[m,p] = 1      ∀ m∈M          (each manager gets one project)
  (2)  Σ_{m∈M}  x[m,p] = 1      ∀ p∈P          (each project gets one manager)
  (3)  x[m,p] ∈ {0,1}           ∀ m∈M, p∈P
```

**Data Mapping**

```
table_id : file_0_view_0

Index Sets
  M  ← distinct values in column "Manager"
  P  ← {1,2,3,4,5,6,7}  (derived from column names "Project i Cost")

Parameter Mapping
  For each row r:
      let m = value of column "Manager" in row r
      For each project i = 1..7:
          let p = i
          c[m,p] = numeric value in column "Project i Cost" of row r
```

The mapping rules above bind every parameter to the corresponding cell in the CSV file, ensuring the model uses the full cost matrix without enumerating individual values.