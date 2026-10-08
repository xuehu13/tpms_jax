"""Optional mechanical stability utilities for the unique Explicit entry."""

def conservative_step_bound(ex,batch_cells=128):
    """Factory: cell-absolute row-sum upper bound for M^-1/2 K M^-1/2.

    Reuses the shared material stress derivative and the force kernel's actual
    reference gradients/weights. No global K/H storage, force projection or
    alteration of physical negative stiffness. Frozen-state frequency only.
    """
    if ex.kernel_geometry is None:raise ValueError('Conservative bound requires compact Cartesian HEX27')
    if batch_cells<1:raise ValueError('Expected positive stability batch size')
    batch=math.gcd(ex.p.fe.num_cells,batch_cells)
    grads=ex.kernel_geometry[1];weights=ex.kernel_geometry[2].reshape(-1)
    cell_ids=ex.ids[ex.device_cells]
    invsqrt=jnp.where(jnp.arange(ex.nc)==ex.pin,0.,jax.lax.rsqrt(ex.mass))
    cell_invsqrt=invsqrt[cell_ids]
    material_tangent=jax.vmap(jax.jacfwd(ex.p.material_stress,argnums=0))
    def bound(q,h):
        Hmacro=jnp.zeros((3,3)).at[2,2].set(h)
        inputs=[q[cell_ids],ex.scale,cell_invsqrt]
        packed=[x.reshape((-1,batch,*x.shape[1:])) for x in inputs]
        def local(data):
            w,scale,m=data
            F=jnp.eye(3)+Hmacro+jnp.einsum('cni,qnj->cqij',w,grads)
            A=material_tangent(F.reshape(-1,3,3),scale.ravel()).reshape((*scale.shape,3,3,3,3))
            K=jnp.einsum('cqijkl,qaj,qbl,q->caibk',A,grads,grads,weights)
            # Sum absolute element contributions BEFORE periodic assembly.
            # Triangle inequality bounds every assembled absolute row sum,
            # including repeated local periodic classes (N=1).
            rows=jnp.sum(jnp.abs(K)*m[:,None,None,:,None],axis=(-2,-1))*m[:,:,None]
            J=jnp.linalg.det(F);required=ex.p.requires_positive_J(scale)
            finite=jnp.all(jnp.isfinite(A))&jnp.all(jnp.isfinite(rows))&jnp.all(jnp.isfinite(J))
            return rows,finite,jnp.min(J),jnp.min(jnp.where(required,J,jnp.inf)),jnp.sum(required&(J<=0)),jnp.sum(required)
        rows,finite,Jmin,NHmin,invalid,required=jax.lax.map(local,packed)
        assembled=jnp.zeros((ex.nc,3)).at[cell_ids.ravel()].add(rows.reshape((-1,3)))
        flat=assembled.ravel();index=jnp.argmax(flat)
        return {'row_sum_bound_s_minus2':jnp.max(flat),'max_periodic_class':index//3,
            'max_component':index%3,'all_material_tangents_finite':jnp.all(finite),
            'J_min':jnp.min(Jmin),'required_positive_J_min':jnp.min(NHmin),
            'invalid_material_points':jnp.sum(invalid),'required_positive_J_points':jnp.sum(required)}
    return jax.jit(bound)


def saved_stability_probe(ex,a,cfg,N):
    """Evaluate retained states only; never advances time or computes design AD."""
    bound=conservative_step_bound(ex,a.stability_batch_cells)
    q0=jnp.zeros((ex.nc,3));started=time.perf_counter()
    warm=bound(q0,0.);jax.block_until_ready(warm);compile_seconds=time.perf_counter()-started
    memory=bound.lower(q0,0.).compile().memory_analysis()
    rows=[]
    for source in (None,*a.stability_states):
        if source is None:q=q0;t=0.;saved_dt=None;label='undeformed'
        else:
            with np.load(source) as f:
                if int(f['N'])!=N:raise ValueError('Retained state mesh mismatch')
                q=jnp.asarray(f['q']);t=float(f['time']);saved_dt=float(f['dt'])
            if q.shape!=q0.shape:raise ValueError('Retained periodic DOF shape mismatch')
            label=str(source)
        s=np.clip(t/a.load_time,0.,1.);h=-.2*(10*s**3-15*s**4+6*s**5)
        costs=[]
        for _ in range(2):
            start=time.perf_counter();values=bound(q,h);jax.block_until_ready(values)
            costs.append(time.perf_counter()-start)
        row={k:(float(v) if np.isfinite(float(v)) else None) for k,v in values.items()}
        row['all_material_tangents_finite']=bool(values['all_material_tangents_finite'])
        for k in ('max_periodic_class','max_component','invalid_material_points','required_positive_J_points'):row[k]=int(values[k])
        R=row['row_sum_bound_s_minus2']
        good=row['all_material_tangents_finite'] and not row['invalid_material_points'] and R is not None and R>0
        limit=.8*2/math.sqrt(R) if good else None
        row.update(source=label,source_sha256=sha(source) if source is not None else None,
            time_s=t,compression=-h,source_dt_seconds=saved_dt,
            nominal_initial_dt_seconds=ex.dt_estimate,bound_seconds=costs,
            safe_dt_seconds=min(ex.dt_estimate,limit) if good else None,
            safe_dt_fraction_of_initial=min(1.,limit/ex.dt_estimate) if good else None)
        rows.append(row);write(a.output/'progress.json',{'role':'saved_state_bound_only','rows':rows})
        print('STABILITY_BOUND '+json.dumps(row),flush=True)
    # Cost reference only: shared force, no velocity/time advance.
    start=time.perf_counter();jax.block_until_ready(ex.force(q0,0.));force_compile=time.perf_counter()-start
    costs=[]
    for _ in range(2):
        start=time.perf_counter();jax.block_until_ready(ex.force(q0,0.));costs.append(time.perf_counter()-start)
    cfg.update(diagnostic_only=True,target_compression=None,stability_bound='cell-absolute periodic mass-normalized row sum',
        stability_safety=.8,stability_batch_cells=a.stability_batch_cells,stability_state_paths=[str(p) for p in a.stability_states])
    write(a.output/'input.json',cfg)
    write(a.output/'result.json',{'status':'saved_state_bound_diagnostic_complete','rows':rows,
        'compile_seconds':compile_seconds,'force_compile_seconds':force_compile,'force_seconds':costs,
        'compiled_bound_temporary_bytes':memory.temp_size_in_bytes if memory is not None else None,
        'peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'scope':'Conservative frozen-state positive-frequency bound and cost. No time advance, design AD, full-path stability or 20% accuracy certification.'})
