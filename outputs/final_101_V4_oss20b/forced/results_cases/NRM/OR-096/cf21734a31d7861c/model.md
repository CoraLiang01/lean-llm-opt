##### Sets
- \(S\): set of schools  
- \(N\): set of neighborhoods  

##### Parameters
- \(\text{capacity}[s]\) from table **file_0_view_0**, column **Capacity**  
- \(\text{pop\_white}[n]\) from table **file_1_view_0**, column **Population_White**  
- \(\text{pop\_nonwhite}[n]\) from table **file_1_view_0**, column **Population_NonWhite**  
- \(\text{dist}[s,n]\) from table **file_2_view_0**, columns **N01**–**N31** (one column per neighborhood)  

##### Variables
- \(x_{\text{white}}[s,n]\) integer, number of white students from neighborhood \(n\) assigned to school \(s\)  
- \(x_{\text{nonwhite}}[s,n]\) integer, number of non‑white students from neighborhood \(n\) assigned to school \(s\)  

##### Objective
\[
\min \sum_{s\in S}\sum_{n\in N}\text{dist}[s,n]\;\bigl(x_{\text{white}}[s,n]+x_{\text{nonwhite}}[s,n]\bigr)
\]

##### Constraints  

1. **School capacity**  
\[
\sum_{n\in N}\bigl(x_{\text{white}}[s,n]+x_{\text{nonwhite}}[s,n]\bigr)\;\le\;\text{capacity}[s],
\qquad \forall s\in S
\]

2. **Neighborhood white supply**  
\[
\sum_{s\in S}x_{\text{white}}[s,n]\;=\;\text{pop\_white}[n],
\qquad \forall n\in N
\]

3. **Neighborhood non‑white supply**  
\[
\sum_{s\in S}x_{\text{nonwhite}}[s,n]\;=\;\text{pop\_nonwhite}[n],
\qquad \forall n\in N
\]

4. **Racial balance (60 % white ±10 %)**  
\[
0.5\;\sum_{n\in N}\bigl(x_{\text{white}}[s,n]+x_{\text{nonwhite}}[s,n]\bigr)
\;\le\;
\sum_{n\in N}x_{\text{white}}[s,n]
\;\le\;
0.7\;\sum_{n\in N}\bigl(x_{\text{white}}[s,n]+x_{\text{nonwhite}}[s,n]\bigr),
\qquad \forall s\in S
\]

5. **Variable domain**  
\[
x_{\text{white}}[s,n],\,x_{\text{nonwhite}}[s,n]\;\in\;\mathbb{Z}_{\ge0},
\qquad \forall s\in S,\;n\in N
\]

##### Data Mapping
| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(\text{capacity}[s]\) | **file_0_view_0** | **Capacity** |
| School names | **file_0_view_0** | **School** |
| \(\text{pop\_white}[n]\) | **file_1_view_0** | **Population_White** |
| \(\text{pop\_nonwhite}[n]\) | **file_1_view_0** | **Population_NonWhite** |
| Neighborhood names | **file_1_view_0** | **Neighborhood** |
| \(\text{dist}[s,n]\) | **file_2_view_0** | **N01**–**N31** (one column per neighborhood) |