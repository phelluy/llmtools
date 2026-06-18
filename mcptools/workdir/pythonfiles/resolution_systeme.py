from sympy import symbols, Matrix, solve, det, simplify

"""
Résolution du système :
    2x + 3y + z = 4
    -x + my + 2z = 5
    7x + 3y + (m-5)z = 7
"""

# Variables
x, y, z, m = symbols('x y z m')

# Matrice des coefficients
A = Matrix([
    [2, 3, 1],
    [-1, m, 2],
    [7, 3, m-5]
])

# Vecteur des constantes
b = Matrix([4, 5, 7])

# Calcul du déterminant
det_A = A.det()
det_A_factorise = det_A.factor()
print(f"Déterminant : det(A) = {det_A_factorise}")

# Valeurs critiques
critiques = solve(det_A, m)
print(f"Valeurs critiques : m ∈ {{{', '.join(str(c) for c in critiques)}}}")

print("\n" + "="*60)
print("CAS 1 : m ≠ 1 et m ≠ 6")
print("="*60)
print("det(A) ≠ 0, le système a une solution unique :")
print(f"    x = (2m - 9)/(m - 1)")
print(f"    y = 7/(m - 1)")
print(f"    z = -7/(m - 1)")

print("\n" + "="*60)
print("CAS 2 : m = 1")
print("="*60)
print("Rang(A) = 2, Rang(matrice augmentée) = 3")
print("Système incompatible. Aucune solution.")

print("\n" + "="*60)
print("CAS 3 : m = 6")
print("="*60)
print("Rang(A) = 2, Rang(matrice augmentée) = 2")
print("Système compatible indeterminate (1 paramètre libre) :")
print(f"    x = 3/5")
print(f"    y = 14/15 - z/3")
print(f"    z libre")
print("\nOu paramétré avec z = t :")
print(f"    x = 3/5")
print(f"    y = 14/15 - t/3")
print(f"    z = t")
print("    t ∈ ℝ")
