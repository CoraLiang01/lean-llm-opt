Mathematical Model

Sets:
  Let 𝒞 be the set of Operations Research courses in courses_42.csv:
    𝒞 = {C22, C23, C24, C25, C26, C27, C28}

Parameters (from table_id: file_0_view_0):
  For each course c ∈ 𝒞:
    credits_c = value in column "credits" for course_id c
    interest_c = value in column "interest_points" for course_id c

Decision Variables:
  For each c ∈ 𝒞:
    x_c ∈ {0,1}   (1 if course c is selected, 0 otherwise)

Objective:
  Maximize total interest points:
    maximize   ∑_{c ∈ 𝒞} interest_c · x_c

Constraint:
  Total credits of selected courses ≤ 20:
    ∑_{c ∈ 𝒞} credits_c · x_c ≤ 20

Variable domains:
  x_c ∈ {0,1}   for all c ∈ 𝒞

Data Mapping:
  - Set 𝒞 and all parameters credits_c, interest_c are taken from table_id: file_0_view_0, columns "course_id", "credits", "interest_points", filtered where "discipline" = "Operations Research".
  - Decision variables x_c correspond to course_id in this filtered set.