"""Periodic distance to a fixed triangle midsurface; physical-thickness occupancy.

Distance queries are host geometry preprocessing. Thickness projection is JAX;
this does not provide a differentiable shape/closest-triangle map.
"""
from itertools import product
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64',True)


def thickness_occupancy(distance, thickness, interface_scale):
    """rho=sigmoid((t/2-d)/ell); one 10%-90% interface width is 2*ln(9)*ell.

    Validate concrete t>0 and ell>0 outside AD. Keep ell fixed when varying t.
    d is unsigned distance, not the amplitude of an implicit function.
    """
    return jax.nn.sigmoid((jnp.asarray(thickness)/2-jnp.asarray(distance))/interface_scale)


class PeriodicSurfaceDistance:
    """Double-precision triangle distance on a cubic period, with reusable AABB.

    The input surface is the one-cell cut in [0,L]^3. A 3x3x3 tiling covers all
    possible nearest images for a wrapped query. Open cut boundaries require
    an independent seam/topology check; wrapping alone cannot repair a gap.
    """
    def __init__(self, vertices, triangles, period=1.):
        import igl
        vertices=np.asarray(vertices,dtype=np.float64)
        triangles=np.asarray(triangles,dtype=np.int64)
        if not np.isfinite(period) or period<=0:
            raise ValueError('Expected finite positive period')
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
            raise ValueError('Expected finite (vertices,3) coordinates')
        if triangles.ndim!=2 or triangles.shape[1]!=3 or not len(triangles):
            raise ValueError('Expected nonempty (triangles,3) connectivity')
        if triangles.min()<0 or triangles.max()>=len(vertices):
            raise ValueError('Triangle index outside vertex range')
        if vertices.min()<-1e-10*period or vertices.max()>period*(1+1e-10):
            raise ValueError('Expected one-cell surface in [0,period]^3')
        self.period=float(period);self.face_count=len(triangles)
        shifts=list(product((-1,0,1),repeat=3))
        self.vertices=np.vstack([vertices+np.asarray(s)*period for s in shifts])
        self.triangles=np.vstack([triangles+i*len(vertices) for i in range(len(shifts))])
        self.tree=igl.AABB();self.tree.init(self.vertices,self.triangles)

    def query(self, points, chunk_size=65536, details=False):
        points=np.asarray(points,dtype=np.float64)
        if points.shape[-1]!=3 or not np.isfinite(points).all():
            raise ValueError('Expected finite (...,3) query points')
        shape=points.shape[:-1];flat=points.reshape(-1,3)%self.period
        distance=np.empty(len(flat));faces=np.empty(len(flat),dtype=np.int64)
        closest=np.empty_like(flat) if details else None
        for start in range(0,len(flat),chunk_size):
            stop=min(start+chunk_size,len(flat))
            squared,index,hit=self.tree.squared_distance(self.vertices,self.triangles,flat[start:stop])
            if np.any(squared<-1e-14) or not np.isfinite(squared).all():
                raise ValueError('Invalid triangle distance result')
            distance[start:stop]=np.sqrt(np.maximum(squared,0.))
            faces[start:stop]=index%self.face_count
            if details:closest[start:stop]=hit
        distance=distance.reshape(shape)
        if details:return distance,faces.reshape(shape),closest.reshape(points.shape)
        return distance
