"""Local, fixed-connectivity geometry for the actual diverse_28 coefficients.

Normalized coordinates and parameter order are (c1, c2, c0):
    f = c1*cos(2*pi*x)*cos(2*pi*z)
        + c2*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) + c0.

The supplied mesh is first projected onto the STRICT baseline zero surface.
Every subsequent geometry solves f(X0 + s*D, p) = 0 again, on fixed lines D.
There is no baseline-residual subtraction and no FD-based derivative rule.
Only the selected smooth root/closest-triangle branch is differentiated.
This module does not construct contact weights or launch any mechanics.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib

import numpy as np
from surface_distance import PeriodicSurfaceDistance


class CoefficientGeometry:
    """Strict implicit vertex continuation, followed by triangle distance.

    ``queries`` and optional ``evaluate(..., query_points=...)`` are fixed
    reference query positions; distance_jac does not include moving queries.
    ``vertices`` is the projected baseline, not the unmodified source mesh.
    Returned array buffers are read-only. At most five evaluations are cached.
    ``need_jac=False`` skips distance_jac, but still returns cheap rim_jac.
    """

    baseline_parameters = np.array([-3.4, 4.6, 2.0], dtype=np.float64)
    parameter_names = ("c1", "c2", "c0")
    boundary_tolerance = 1e-10
    root_tolerance = 1e-12
    denominator_tolerance = 1e-10
    max_line_shift = 0.05  # normalized local branch, not a parameter fit knob
    query_chunk_size = 65536

    def __init__(self, surface_path, queries):
        self.surface_path = str(Path(surface_path))
        with np.load(surface_path) as z:
            raw = np.asarray(z["surface_vertices"], dtype=np.float64).copy()
            self.triangles = np.asarray(z["surface_triangles"], dtype=np.int64).copy()
        if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
            raise ValueError("Expected finite (Nv,3) source vertices")
        if (self.triangles.ndim != 2 or self.triangles.shape[1] != 3
                or not len(self.triangles) or self.triangles.min() < 0
                or self.triangles.max() >= len(raw)):
            raise ValueError("Invalid source triangle connectivity")
        if raw.min() < -self.boundary_tolerance or raw.max() > 1+self.boundary_tolerance:
            raise ValueError("Expected the one-cell cut mesh in [0,1]^3")
        self.raw_vertices = raw.copy()
        self.queries = self._queries(queries).copy()
        self._low = np.isclose(raw, 0., rtol=0., atol=self.boundary_tolerance)
        self._high = np.isclose(raw, 1., rtol=0., atol=self.boundary_tolerance)
        self._locked = self._low | self._high
        anchor = raw.copy()
        anchor[self._low] = 0.
        anchor[self._high] = 1.
        self.rim_vertex_ids = np.r_[np.flatnonzero(self._low[:, 2]),
                                    np.flatnonzero(self._high[:, 2])].astype(np.int64)
        self.rim_side = np.r_[np.zeros(np.count_nonzero(self._low[:, 2]), dtype=np.int64),
                              np.ones(np.count_nonzero(self._high[:, 2]), dtype=np.int64)]
        self.rim_counts = (int(np.count_nonzero(self._low[:, 2])),
                           int(np.count_nonzero(self._high[:, 2])))

        # A fixed constrained normal is also a fixed vertex parameterization.
        # On one/two cut faces it lies in the corresponding face/edge.
        basis, grad = self._basis_grad(anchor, self.baseline_parameters)
        direction = np.where(self._locked, 0., grad)
        length = np.linalg.norm(direction, axis=1)
        if np.any(length <= self.denominator_tolerance):
            bad = np.flatnonzero(length <= self.denominator_tolerance)[:10]
            raise ValueError("No regular constrained-normal continuation at vertices " + str(bad.tolist()))
        self.directions = direction / length[:, None]
        # Periodic aliases evaluate at the identical wrapped coordinate. Their
        # root lines must agree; no independent movement of paired seams.
        canonical = anchor.copy()
        canonical[self._high] = 0.
        _, self._representatives, self._groups = np.unique(
            np.round(canonical, 9), axis=0, return_index=True, return_inverse=True)
        self._periodic_missing = []
        for axis in range(3):
            low_groups = np.unique(self._groups[self._low[:, axis]])
            high_groups = np.unique(self._groups[self._high[:, axis]])
            self._periodic_missing.append(int(np.setxor1d(low_groups, high_groups).size))
        if any(self._periodic_missing):
            raise ValueError("Source cut mesh lacks paired periodic face vertices: " + str(self._periodic_missing))
        d_ref = self.directions[self._representatives[self._groups]]
        direction_gap = float(np.max(np.linalg.norm(self.directions-d_ref, axis=1)))
        if direction_gap > 1e-8:
            raise ValueError("Periodic constrained-normal directions disagree")
        self.directions = d_ref.copy()

        projected, root_audit = self._roots(anchor, self.baseline_parameters)
        self.vertices = projected
        # The initial projection used normals of the supplied approximation.
        # Freeze subsequent continuation directions at the STRICT baseline.
        _, projected_grad = self._basis_grad(projected, self.baseline_parameters)
        projected_direction = np.where(self._locked, 0., projected_grad)
        projected_length = np.linalg.norm(projected_direction, axis=1)
        if np.any(projected_length <= self.denominator_tolerance):
            raise ValueError("Strict baseline has a singular constrained normal")
        projected_direction /= projected_length[:, None]
        self.directions = projected_direction[self._representatives[self._groups]].copy()
        strict_den = np.einsum("vi,vi->v", projected_grad, self.directions)
        if np.min(np.abs(strict_den)) <= self.denominator_tolerance:
            raise ValueError("Strict baseline continuation Jacobian is singular")
        self._reference_cross = self._cross(self.vertices)
        mesh = self._mesh_audit(self.vertices, reference_cross=self._cross(anchor))
        boundary = self._boundary_audit(self.vertices)
        displacement = np.linalg.norm(self.vertices-raw, axis=1)
        raw_basis, _ = self._basis_grad(raw, self.baseline_parameters)
        self.baseline_audit = dict(
            source_path=self.surface_path,
            source_sha256=hashlib.sha256(Path(surface_path).read_bytes()).hexdigest(),
            vertex_count=int(len(raw)), triangle_count=int(len(self.triangles)),
            raw_max_abs_f=float(np.max(np.abs(raw_basis @ self.baseline_parameters))),
            baseline_projection="strict f=0; constrained-normal fixed lines; no residual compensation",
            baseline_projection_max_normalized=float(displacement.max()),
            baseline_projection_rms_normalized=float(np.sqrt(np.mean(displacement**2))),
            baseline_projected_max_abs_f=root_audit["root_max_abs_f"],
            baseline_direction_periodic_gap=direction_gap,
            baseline_strict_direction_min_abs_denominator=float(np.min(np.abs(strict_den))),
            baseline_direction_definition="constrained unit normal at the strict projected baseline",
            rim_counts=list(self.rim_counts), **mesh, **boundary, **root_audit)
        self._cache = OrderedDict()
        for array in (self.raw_vertices, self.vertices, self.triangles,
                      self.directions, self.queries, self.rim_vertex_ids, self.rim_side):
            array.setflags(write=False)

    @staticmethod
    def _queries(points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Expected finite query_points with shape (Nq,3)")
        if not len(points):
            raise ValueError("Empty distance query set")
        return np.ascontiguousarray(points)

    @staticmethod
    def _basis_grad(vertices, parameters):
        # Wrapping makes the numerical evaluations of periodic aliases equal.
        xyz = 2*np.pi*(np.asarray(vertices, dtype=np.float64) % 1.)
        sx, sy, sz = np.sin(xyz).T
        cx, cy, cz = np.cos(xyz).T
        basis = np.column_stack((cx*cz, sx*sy*sz, np.ones(len(xyz))))
        c1, c2, _ = parameters
        grad = 2*np.pi*np.column_stack((
            -c1*sx*cz+c2*cx*sy*sz,
            c2*sx*cy*sz,
            -c1*cx*sz+c2*sx*sy*cz))
        return basis, grad

    def _roots(self, anchor, parameters):
        """Damped nonlinear Newton solve of each scalar line root, in float64."""
        s = np.zeros(len(anchor), dtype=np.float64)
        minimum_denominator = np.inf
        iterations = 0
        for iteration in range(60):
            vertices = anchor+s[:, None]*self.directions
            basis, grad = self._basis_grad(vertices, parameters)
            residual = basis @ parameters
            den = np.einsum("vi,vi->v", grad, self.directions)
            minimum_denominator = min(minimum_denominator, float(np.min(np.abs(den))))
            active = np.abs(residual) > self.root_tolerance
            if not np.any(active):
                break
            if np.any(np.abs(den[active]) <= self.denominator_tolerance):
                raise ValueError("Singular constrained line root denominator")
            step = np.zeros_like(s)
            step[active] = np.clip(-residual[active]/den[active], -.01, .01)
            accepted = ~active
            proposed = s.copy()
            # Per-vertex backtracking; no root iteration is used as AD logic.
            for backtrack in range(16):
                factor = 2.**(-backtrack)
                candidate = s+factor*step
                trial, _ = self._basis_grad(anchor+candidate[:, None]*self.directions, parameters)
                trial_residual = np.abs(trial @ parameters)
                take = (~accepted & (np.abs(candidate) <= self.max_line_shift)
                        & ((trial_residual < np.abs(residual)*(1-1e-4*factor))
                           | (trial_residual <= self.root_tolerance)))
                proposed[take] = candidate[take]
                accepted |= take
                if np.all(accepted):
                    break
            if not np.all(accepted):
                bad = np.flatnonzero(~accepted)[:10]
                raise ValueError("Nonlinear local root failed/backtracking exceeded branch at " + str(bad.tolist()))
            s = proposed
            iterations = iteration+1
        vertices = anchor+s[:, None]*self.directions
        basis, grad = self._basis_grad(vertices, parameters)
        residual = basis @ parameters
        den = np.einsum("vi,vi->v", grad, self.directions)
        if np.max(np.abs(residual)) > self.root_tolerance:
            raise ValueError("Nonlinear root residual did not converge")
        if np.min(np.abs(den)) <= self.denominator_tolerance:
            raise ValueError("Root Jacobian is singular")
        if vertices.min() < -1e-10 or vertices.max() > 1+1e-10:
            raise ValueError("Continued vertices leave the fixed one-cell cut topology")
        return vertices, dict(root_max_abs_f=float(np.max(np.abs(residual))),
                              minJacden=float(np.min(np.abs(den))),
                              root_iteration_min_abs_denominator=float(minimum_denominator),
                              root_iterations=int(iterations),
                              root_max_abs_line_shift=float(np.max(np.abs(s))))

    def _cross(self, vertices):
        triangle = vertices[self.triangles]
        return np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])

    def _mesh_audit(self, vertices, reference_cross=None):
        triangle = vertices[self.triangles]
        cross = self._cross(vertices)
        double_area = np.linalg.norm(cross, axis=1)
        edges = np.stack((triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 1],
                          triangle[:, 0]-triangle[:, 2]), axis=1)
        lengths = np.linalg.norm(edges, axis=2)
        if np.any(double_area <= 1e-16) or not np.isfinite(double_area).all():
            raise ValueError("Degenerate continued surface triangle")
        flips = (int(np.count_nonzero(np.einsum("fi,fi->f", cross, reference_cross) <= 0.))
                 if reference_cross is not None else 0)
        if flips:
            raise ValueError("Continued surface triangle reverses relative orientation")
        return dict(min_triangle_area_normalized=float(.5*double_area.min()),
                    min_edge_length_normalized=float(lengths.min()),
                    max_longest_edge_squared_over_double_area=float(
                        np.max(lengths.max(axis=1)**2/double_area)),
                    total_surface_area_normalized=float(.5*double_area.sum()),
                    triangle_orientation_reversals=flips)

    def _boundary_audit(self, vertices):
        locked_error = np.r_[np.abs(vertices[self._low]), np.abs(vertices[self._high]-1.)]
        boundary_error = float(locked_error.max()) if len(locked_error) else 0.
        canonical = vertices.copy()
        canonical[self._high] -= 1.
        paired = canonical[self._representatives[self._groups]]
        seam_error = float(np.max(np.linalg.norm(canonical-paired, axis=1)))
        if boundary_error > 1e-12 or seam_error > 1e-8:
            raise ValueError("Continued boundary/periodic seam constraints failed")
        return dict(max_locked_boundary_coordinate_error=boundary_error,
                    max_periodic_vertex_gap_normalized=seam_error,
                    periodic_unpaired_boundary_groups=list(self._periodic_missing),
                    boundary_coordinates_fixed=True)

    def evaluate(self, p, query_points=None, need_jac=True):
        """Re-solve real nonlinear geometry and return exact branch Jacobians.

        distance_jac axes: (query, coefficient).
        rim_jac axes: (rim point, spatial coordinate, coefficient).
        Rim order: source vertex IDs at z=0, then source IDs at z=1.
        """
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Expected finite parameters (c1,c2,c0)")
        queries = self.queries if query_points is None else self._queries(query_points)
        query_digest = hashlib.sha256(memoryview(np.ascontiguousarray(queries)).cast("B")).hexdigest()
        key = (tuple(p.tolist()), queries.shape, query_digest, bool(need_jac))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        vertices, root_audit = self._roots(self.vertices, p)
        basis, grad = self._basis_grad(vertices, p)
        den = np.einsum("vi,vi->v", grad, self.directions)
        # Implicit line-root derivative; no finite differences are involved.
        vertex_jac = -self.directions[:, :, None]*basis[:, None, :]/den[:, None, None]
        mesh = self._mesh_audit(vertices, self._reference_cross)
        boundary = self._boundary_audit(vertices)
        tree = PeriodicSurfaceDistance(vertices, self.triangles)
        distance = np.empty(len(queries), dtype=np.float64)
        jacobian = np.empty((len(queries), 3), dtype=np.float64) if need_jac else None
        faces = np.empty(len(queries), dtype=np.int64)
        min_bary, max_bary, min_gram = np.inf, -np.inf, np.inf
        zero_distance_count = 0
        for first in range(0, len(queries), self.query_chunk_size):
            last = min(first+self.query_chunk_size, len(queries))
            query = queries[first:last] % 1.
            squared, index, hit = tree.tree.squared_distance(tree.vertices, tree.triangles, query)
            squared = np.asarray(squared, dtype=np.float64).ravel()
            index = np.asarray(index, dtype=np.int64).ravel()
            hit = np.asarray(hit, dtype=np.float64)
            if not np.isfinite(squared).all() or np.min(squared) < -1e-14:
                raise ValueError("Invalid periodic closest-triangle distance")
            d = np.sqrt(np.maximum(squared, 0.))
            triangle = tree.vertices[tree.triangles[index]]
            e0, e1, h = triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0], hit-triangle[:, 0]
            dot = lambda a, b: np.einsum("ni,ni->n", a, b)
            d00, d01, d11 = dot(e0, e0), dot(e0, e1), dot(e1, e1)
            gram = d00*d11-d01*d01
            if np.any(gram <= 0.) or not np.isfinite(hit).all():
                raise ValueError("Degenerate closest triangle")
            b1 = (d11*dot(h, e0)-d01*dot(h, e1))/gram
            b2 = (d00*dot(h, e1)-d01*dot(h, e0))/gram
            bary = np.column_stack((1-b1-b2, b1, b2))
            if bary.min() < -1e-7 or bary.max() > 1+1e-7:
                raise ValueError("Invalid closest-point barycentric coordinates")
            base_face = index % len(self.triangles)
            distance[first:last] = d
            faces[first:last] = base_face
            zero_distance_count += int(np.count_nonzero(d <= 1e-14))
            if need_jac:
                if np.any(d <= 1e-14):
                    raise ValueError("Unsigned distance Jacobian undefined at a zero-distance query")
                closest_jac = np.einsum("ni,nijk->njk", bary,
                                        vertex_jac[self.triangles[base_face]])
                jacobian[first:last] = -np.einsum("ni,nij->nj", (query-hit)/d[:, None], closest_jac)
            min_bary = min(min_bary, float(bary.min()))
            max_bary = max(max_bary, float(bary.max()))
            min_gram = min(min_gram, float(gram.min()))

        audit = dict(self.baseline_audit)
        audit.update(root_audit)
        audit.update(mesh)
        audit.update(boundary)
        audit.update(parameters=p.tolist(), parameter_names=list(self.parameter_names),
                     root_family="strict f(X0+s*D,p)=0; fixed constrained-normal D and triangle connectivity",
                     min_distance_normalized=float(distance.min()),
                     closest_min_bary=min_bary, closest_max_bary=max_bary,
                     closest_min_triangle_gram_determinant=min_gram,
                     zero_distance_queries=zero_distance_count,
                     selected_branch_only=True, nearest_point_uniqueness_certified=False,
                     AD_rule_uses_FD=False, contact_assignment_or_interpolation_included=False,
                     max_uniform_coefficient_scaling_vertex_null=float(np.max(np.abs(
                         np.einsum("vik,k->vi", vertex_jac, p)))))
        result = dict(distance=distance, distance_jac=jacobian,
                      rim_points=vertices[self.rim_vertex_ids].copy(),
                      rim_jac=vertex_jac[self.rim_vertex_ids].copy(),
                      closest_triangle=faces, audit=audit)
        for value in result.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self._cache[key] = result
        while len(self._cache) > 5:
            self._cache.popitem(last=False)
        return result
