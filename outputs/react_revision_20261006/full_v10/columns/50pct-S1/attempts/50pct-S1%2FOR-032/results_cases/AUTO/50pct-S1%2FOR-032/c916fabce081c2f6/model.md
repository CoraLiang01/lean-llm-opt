Mathematical Model

Sets:
  Let 𝒞 be the set of Operations Research courses in courses_42.csv with discipline = "Operations Research".
    𝒞 = {C22, C23, C24, C25, C26, C27, C28}

Parameters (from file_0_view_0):
  For each course c ∈ 𝒞:
    credits_c: credits of course c (column "credits")
    interest_c: interest points of course c (column "interest_points")

Decision Variables:
  For each c ∈ 𝒞:
    x_c ∈ {0,1}    (1 if course c is selected, 0 otherwise)

Objective:
  Maximize total interest points:
    maximize   ∑_{c ∈ 𝒞} interest_c * x_c

Constraint:
  Total credits of selected courses ≤ 20:
    ∑_{c ∈ 𝒞} credits_c * x_c ≤ 20

Variable domains:
  x_c ∈ {0,1}   for all c ∈ 𝒞

Data Mapping:
  𝒞, credits_c, interest_c are defined by the rows and columns of file_0_view_0 (courses_42.csv, filtered to discipline = "Operations Research", columns: course_id, credits, interest_points).