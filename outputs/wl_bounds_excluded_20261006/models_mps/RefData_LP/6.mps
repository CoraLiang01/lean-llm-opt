* Signature: 0x74b58b452f0412a5
NAME _copy
OBJSENSE MAX
ROWS
 N  OBJ
 L  Material_Availability[1]
 L  Material_Availability[2]
 L  Material_Availability[3]
 G  ProdA_Spec1
 L  ProdA_Spec2
 G  ProdB_Spec1
 L  ProdB_Spec2
COLUMNS
    x[A,1]    OBJ       -5
    x[A,1]    Material_Availability[1]  1
    x[A,1]    ProdA_Spec1  0.5
    x[A,1]    ProdA_Spec2  -0.25
    x[A,2]    OBJ       35
    x[A,2]    Material_Availability[2]  1
    x[A,2]    ProdA_Spec1  -0.5
    x[A,2]    ProdA_Spec2  0.75
    x[A,3]    OBJ       25
    x[A,3]    Material_Availability[3]  1
    x[A,3]    ProdA_Spec1  -0.5
    x[A,3]    ProdA_Spec2  -0.25
    x[B,1]    OBJ       5
    x[B,1]    Material_Availability[1]  1
    x[B,1]    ProdB_Spec1  0.75
    x[B,1]    ProdB_Spec2  -0.5
    x[B,2]    OBJ       45
    x[B,2]    Material_Availability[2]  1
    x[B,2]    ProdB_Spec1  -0.25
    x[B,2]    ProdB_Spec2  0.5
    x[B,3]    OBJ       35
    x[B,3]    Material_Availability[3]  1
    x[B,3]    ProdB_Spec1  -0.25
    x[B,3]    ProdB_Spec2  -0.5
    x[C,1]    OBJ       -5
    x[C,1]    Material_Availability[1]  1
    x[C,2]    OBJ       35
    x[C,2]    Material_Availability[2]  1
    x[C,3]    OBJ       25
    x[C,3]    Material_Availability[3]  1
RHS
    RHS1      Material_Availability[1]  100
    RHS1      Material_Availability[2]  100
    RHS1      Material_Availability[3]  60
BOUNDS
ENDATA
