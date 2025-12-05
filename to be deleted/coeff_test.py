import numpy as np
from scipy.linalg import solve
import pandas as pd

# Define the coefficient matrix A and the constants vector b
# System of equations:
# 13.503x + 14.93y + 16.88z + 17.5i = 16.27
# x + y + z + i = 1
# 15.55x + 15.503y + 16.529z + 19i = 17.016
# 16x + 15.39y + 15.547z + 16.75i = 15.963

A = np.array([
    [13.503, 14.93, 16.88, 17.5],
    [1, 1, 1, 1],
    [15.55, 15.503, 16.529, 19],
    [16, 15.39, 15.547, 16.75]
])

b = np.array([16.27, 1, 17.016, 15.963])

print("Coefficient Matrix A:")
print(A)
print("\nConstants vector b:")
print(b)

# Check if the matrix is singular (determinant = 0)
det_A = np.linalg.det(A)
print(f"\nDeterminant of A: {det_A:.10f}")

if abs(det_A) < 1e-10:
    print("Warning: Matrix is nearly singular - solution may be unstable")

# Solve the system using scipy's solve function
try:
    solution = solve(A, b)
    print("\nSolution:")
    print(f"x = {solution[0]:.10f}")
    print(f"y = {solution[1]:.10f}")
    print(f"z = {solution[2]:.10f}")
    print(f"i = {solution[3]:.10f}")

    # Verify the solution by substituting back
    print("\n" + "=" * 50)
    print("VERIFICATION - Substituting solution back into equations:")
    print("=" * 50)

    for eq_num in range(4):
        result = np.dot(A[eq_num], solution)
        error = abs(result - b[eq_num])
        print(f"Equation {eq_num + 1}: {result:.10f} = {b[eq_num]:.3f} (Error: {error:.2e})")

    # Check constraint x + y + z + i = 1
    sum_check = sum(solution)
    print(f"\nConstraint check (x + y + z + i): {sum_check:.10f} should equal 1.0")
    print(f"Sum constraint error: {abs(sum_check - 1.0):.2e}")

    # Additional analysis
    print("\n" + "=" * 50)
    print("ADDITIONAL ANALYSIS:")
    print("=" * 50)

    # Condition number
    cond_num = np.linalg.cond(A)
    print(f"Condition number: {cond_num:.2f}")
    if cond_num > 1000:
        print("Warning: High condition number indicates the system may be ill-conditioned")

    # Check if all solutions are reasonable (between 0 and 1 for proportion-like problems)
    print(f"\nSolution ranges:")
    for i, var in enumerate(['x', 'y', 'z', 'i']):
        val = solution[i]
        print(f"{var}: {val:.6f} ({'✓' if 0 <= val <= 1 else '⚠️ outside [0,1]'})")

except np.linalg.LinAlgError as e:
    print(f"Error solving the system: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

# Alternative method using numpy's lstsq for comparison
print("\n" + "=" * 50)
print("ALTERNATIVE SOLUTION (Least Squares):")
print("=" * 50)

try:
    solution_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("Least squares solution:")
    for i, var in enumerate(['x', 'y', 'z', 'i']):
        print(f"{var} = {solution_lstsq[i]:.10f}")

    if len(residuals) > 0:
        print(f"\nSum of squared residuals: {residuals[0]:.2e}")
    print(f"Rank of matrix: {rank}")
    print(f"Singular values: {s}")

except Exception as e:
    print(f"Error with least squares method: {e}")