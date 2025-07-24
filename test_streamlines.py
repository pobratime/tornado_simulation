import numpy as np
import pyvista as pv

# Quick test to see what streamlines returns
x = np.linspace(-0.3, 0.3, 10)
y = np.linspace(-0.3, 0.3, 10)
z = np.linspace(-0.2, 0.2, 8)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create some dummy vectors
vectors = np.zeros((X.size, 3))
for i in range(X.size):
    xi, yi, zi = X.flat[i], Y.flat[i], Z.flat[i]
    vectors[i] = [-yi, xi, 0]  # Simple rotation

grid = pv.StructuredGrid()
grid.points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
grid.dimensions = X.shape
grid['vectors'] = vectors

print("Testing streamlines method...")
stream = grid.streamlines('vectors', source_center=(0, 0, 0), n_points=10)
print(f"Type of stream: {type(stream)}")

if isinstance(stream, tuple):
    print(f"Stream is a tuple with {len(stream)} elements")
    for i, elem in enumerate(stream):
        print(f"  Element {i}: {type(elem)}")
        if hasattr(elem, 'tube'):
            print(f"    Has tube method: Yes")
        else:
            print(f"    Has tube method: No")
else:
    print(f"Stream has tube method: {hasattr(stream, 'tube')}")
