"""One periodic, fixed-rim shape coordinate; exact closest-triangle JVP.

Only geometry is new. Material, mass projection, contact and time integration
remain in DesignPath. This is a deformed-G surface family, not a claim that
arbitrary remeshing, topology changes or all analytic TPMS weights are smooth.
"""
import hashlib
from functools import lru_cache
from types import SimpleNamespace
from design import DesignPath, B, R, jax, jnp, np
from surface_distance import PeriodicSurfaceDistance


class ShapeGeometry:
    def __init__(self):
        import json
        self.root = B / 'input/G_shape'
        self.protocol = json.loads((self.root / 'geometry_protocol.json').read_text())
        for rel, expected in self.protocol['protected_sources'].items():
            if hashlib.sha256((R / rel).read_bytes()).hexdigest() != expected:
                raise ValueError('Shape geometry source changed: ' + rel)
        with np.load(R / self.protocol['surface_path']) as z:
            self.vertices = z['surface_vertices'].copy()
            self.triangles = z['surface_triangles'].copy()
        with np.load(self.root / 'queries.npz') as z:
            self.gauss_shape = tuple(z['gauss_shape'])
            self.queries = z['points'].copy()
        x, y, z = self.vertices.T
        self.velocity = np.zeros_like(self.vertices)
        self.velocity[:, 0] = np.sin(2*np.pi*x)*np.sin(np.pi*z)**2
        boundary = (np.isclose(x, 0, atol=1e-12, rtol=0) |
                    np.isclose(x, 1, atol=1e-12, rtol=0) |
                    np.isclose(z, 0, atol=1e-12, rtol=0) |
                    np.isclose(z, 1, atol=1e-12, rtol=0))
        self.velocity[boundary] = 0.
        self.n_gauss = int(np.prod(self.gauss_shape))
        self._cached = lru_cache(maxsize=5)(self._query)
        output = jax.ShapeDtypeStruct((len(self.queries),), jnp.float64)

        @jax.custom_jvp
        def distance(a):
            return jax.pure_callback(lambda a: self.evaluate(float(a))[0], output, a)

        @distance.defjvp
        def distance_jvp(primals, tangents):
            a, = primals
            da, = tangents
            d, dd = jax.pure_callback(
                lambda a: self.evaluate(float(a))[:2], (output, output), a)
            # This expression is linear in the tangent and can be transposed.
            return d, dd * da

        self.distance = distance

    def evaluate(self, a):
        if abs(a) > self.protocol['parameter_range'][1] + 1e-15:
            raise ValueError('Outside the frozen local shape family')
        return self._cached(float(a))

    def _query(self, a):
        moved = self.vertices + a*self.velocity
        tree = PeriodicSurfaceDistance(moved, self.triangles)
        d = np.empty(len(self.queries)); dd = np.empty_like(d)
        face = np.empty(len(d), dtype=np.int64)
        min_bary, max_bary, min_det = np.inf, -np.inf, np.inf
        for first in range(0, len(d), 65536):
            last = min(first+65536, len(d))
            query = self.queries[first:last] % 1.
            squared, index, hit = tree.tree.squared_distance(tree.vertices, tree.triangles, query)
            index = np.asarray(index).ravel(); hit = np.asarray(hit)
            distance = np.sqrt(np.maximum(np.asarray(squared).ravel(), 0.))
            tri = tree.vertices[tree.triangles[index]]
            e0, e1, v = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], hit-tri[:, 0]
            dot = lambda u, v: np.einsum('ij,ij->i', u, v)
            d00, d01, d11 = dot(e0, e0), dot(e0, e1), dot(e1, e1)
            det = d00*d11-d01*d01
            if np.any(det <= 0) or np.any(distance <= 1e-14):
                raise ValueError('Degenerate closest triangle or zero unsigned distance')
            b1 = (d11*dot(v,e0)-d01*dot(v,e1))/det
            b2 = (d00*dot(v,e1)-d01*dot(v,e0))/det
            bary = np.column_stack([1-b1-b2, b1, b2])
            base_face = index % len(self.triangles)
            speed = np.einsum('ni,nij->nj', bary, self.velocity[self.triangles[base_face]])
            d[first:last] = distance
            dd[first:last] = -dot((query-hit)/distance[:,None], speed)
            face[first:last] = base_face
            min_bary = min(min_bary, float(bary.min()))
            max_bary = max(max_bary, float(bary.max()))
            min_det = min(min_det, float(det.min()))
        if min_bary < -1e-7 or max_bary > 1+1e-7:
            raise ValueError('Invalid closest-point barycentric coordinates')
        audit = dict(min_distance=float(d.min()), min_bary=min_bary,
                     max_bary=max_bary, min_triangle_gram_determinant=min_det)
        return d, dd, face, audit


class ShapeDesignPath(DesignPath):
    def __init__(self, config_path='input/protocol.json', geometry_path='input/design_geometry.npz'):
        super().__init__(config_path, geometry_path)
        self.shape_geometry = ShapeGeometry()

    def _fields(self, p):
        # The two-parameter case is only used by the inherited baseline check.
        a = p[2] if p.shape[0] == 3 else jnp.array(0., dtype=p.dtype)
        geometry = self.shape_geometry
        d = geometry.distance(a)
        proxy = SimpleNamespace(ex=self.ex,
            distance=d[:geometry.n_gauss].reshape(geometry.gauss_shape),
            cap_distance=d[geometry.n_gauss:],
            cap_raw_area=self.cap_raw_area, assignment=self.assignment)
        return DesignPath._fields(proxy, p[:2])
