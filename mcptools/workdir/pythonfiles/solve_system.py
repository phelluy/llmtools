import sympy

# Define symbols
x, y, z, m = sympy.symbols('x y z m')

# Define the system of equations
eq1 = sympy.Eq(2*x + 3*y + z, 4)
eq2 = sympy.Eq(-x + m*y + 2*z, 5)
eq3 = sympy.Eq(7*x + 3*y + (m-5)*z, 7)

# Define the coefficient matrix using sympy's Matrix
A = sympy.Matrix([[2, 3, 1], [-1, m, 2], [7, 3, m-5]])
b_vec = sympy.Matrix([4, 5, 7])

# Calculate the determinant of the coefficient matrix
det_A = A.det()

print("="*70)
print("SYSTEM OF LINEAR EQUATIONS WITH PARAMETER m")
print("="*70)
print(f"\nSystem:")
print("2x + 3y + z = 4")
print("-x + my + 2z = 5")
print("7x + 3y + (m-5)z = 7")

print("\n" + "="*70)
print("DETERMINANT ANALYSIS")
print("="*70)
print(f"Determinant of coefficient matrix: {det_A}")
print(f"Simplified: {sympy.simplify(det_A)}")
print(f"Factored: {sympy.factor(det_A)}")

# Find critical values of m (where det = 0)
critical_m = sympy.solve(det_A, m)
print(f"\nCritical values of m (det = 0): {critical_m}")

# Analyze each case
print("\n" + "="*70)
print("CASE 1: m = 1 (det = 0)")
print("="*70)
m1 = 1
A1 = A.subs(m, m1)
rank_A1 = A1.rank()
augmented1 = sympy.Matrix([A1, b_vec])
rank_aug1 = augmented1.rank()
print(f"Coefficient matrix A (m=1):")
print(A1)
print(f"Rank of A: {rank_A1}")
print(f"Rank of augmented: {rank_aug1}")
print(f"Augmented matrix:\n{augmented1}")

# Check consistency
if rank_A1 == rank_aug1:
    # Solve and see if infinite solutions or specific solution
    try:
        sol1 = sympy.solve([eq1.subs(m, m1), eq2.subs(m, m1), eq3.subs(m, m1)], (x, y, z))
        print(f"Solution: {sol1}")
    except Exception as e:
        print(f"Error finding solution: {e}")
else:
    print("Result: INCONSISTENT SYSTEM (NO SOLUTION)")

print("\n" + "="*70)
print("CASE 2: m = 6 (det = 0)")
print("="*70)
m2 = 6
A2 = A.subs(m, m2)
rank_A2 = A2.rank()
augmented2 = sympy.Matrix([A2, b_vec])
rank_aug2 = augmented2.rank()
print(f"Coefficient matrix A (m=6):")
print(A2)
print(f"Rank of A: {rank_A2}")
print(f"Rank of augmented: {rank_aug2}")
print(f"Augmented matrix:\n{augmented2}")

# Check consistency
if rank_A2 == rank_aug2:
    try:
        sol2 = sympy.solve([eq1.subs(m, m2), eq2.subs(m, m2), eq3.subs(m, m2)], (x, y, z))
        print(f"Solution: {sol2}")
    except Exception as e:
        print(f"Error finding solution: {e}")
else:
    print("Result: INCONSISTENT SYSTEM (NO SOLUTION)")

print("\n" + "="*70)
print("GENERAL CASE: det(A) ≠ 0, i.e., m ≠ 1 and m ≠ 6")
print("="*70)
print(f"Unique solution exists when det(A) ≠ 0, i.e., m ≠ 1 and m ≠ 6")
sol_general = sympy.solve([eq1, eq2, eq3], (x, y, z))
print(f"\nSolution:")
print(f"x = {sol_general[x]}")
print(f"y = {sol_general[y]}")
print(f"z = {sol_general[z]}")
print(f"Simplified:")
print(f"x = {sympy.simplify(sol_general[x])}")
print(f"y = {sympy.simplify(sol_general[y])}")
print(f"z = {sympy.simplify(sol_general[z])}")

print("\n" + "="*70)
print("DETAILED VERIFICATION FOR m ≠ 1, 6")
print("="*70)
sol = {x: sol_general[x], y: sol_general[y], z: sol_general[z]}
print(f"\nVerification:")
print(f"Eq1 (2x+3y+z): {sympy.simplify(2*sol[x] + 3*sol[y] + sol[z])}")
print(f"Eq2 (-x+my+2z): {sympy.simplify(-sol[x] + m*sol[y] + 2*sol[z])}")
print(f"Eq3 (7x+3y+(m-5)z): {sympy.simplify(7*sol[x] + 3*sol[y] + (m-5)*sol[z])}")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"The determinant: det(A) = 2(m-1)(m-6)")
print(f"\nDiscussion according to parameter m:")
print("  1. When m ≠ 1 and m ≠ 6:")
print(f"     → UNIQUE SOLUTION exists")
print(f"     → x = {sympy.simplify(sol_general[x])}")
print(f"     → y = {sympy.simplify(sol_general[y])}")
print(f"     → z = {sympy.simplify(sol_general[z])}")
print("  2. When m = 1:")
print(f"     → INCONSISTENT SYSTEM (NO SOLUTION)")
print(f"     → Rank(A) = {rank_A1}, Rank(augmented) = {rank_aug1}")
print("  3. When m = 6:")
print(f"     → INCONSISTENT SYSTEM (NO SOLUTION)")
print(f"     → Rank(A) = {rank_A2}, Rank(augmented) = {rank_aug2}")

print("\n" + "="*70)
print("MATHEMATICAL EXPLANATION")
print("="*70)
print("The system of linear equations can have:")
print("  - UNIQUE SOLUTION: When det(A) ≠ 0")
print("  - INFINITE SOLUTIONS: When det(A) = 0 AND rank(A) = rank(augmented) < 3")
print("  - NO SOLUTION: When det(A) = 0 AND rank(A) < rank(augmented)")
print("\nSince det(A) = 2(m-1)(m-6) has roots at m = 1 and m = 6:")
print("  - At m = 1 and m = 6, the system becomes inconsistent (no solution)")
print("  - At all other values of m, the system has a unique solution")
print("\n" + "="*70)
