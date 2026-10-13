def __post_init__(self):

    if type(self.mesh) != type([]):
        self.mesh = [self.mesh]
        self.vec = [self.vec]
        self.ele_type = [self.ele_type]
        self.quadrature_rule = [self.quadrature_rule]
        self.quadrature_order = [self.quadrature_order]
        self.dirichlet_bc_info = [self.dirichlet_bc_info]

    self.num_vars = len(self.mesh)

    self.fes = [FiniteElement(mesh=self.mesh[i], 
                              vec=self.vec[i], 
                              dim=self.dim, 
                              ele_type=self.ele_type[i], 
                              quadrature_rule=self.quadrature_rule[i] if type(self.quadrature_rule) == type([]) else self.quadrature_rule,
                              quadrature_order=self.quadrature_order[i] if type(self.quadrature_order) == type([]) else self.quadrature_order,
                              dirichlet_bc_info=self.dirichlet_bc_info[i] if type(self.dirichlet_bc_info) == type([]) else self.dirichlet_bc_info) \
                for i in range(self.num_vars)] 

    self.cells_list = [fe.cells for fe in self.fes]
    # Assume all fes have the same number of cells, same dimension
    self.num_cells = self.fes[0].num_cells
    self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(self.location_fns)

    self.offset = [0] 
    for i in range(len(self.fes) - 1):
        self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

    def find_ind(*x):
        inds = []
        for i in range(len(x)):
            crt_ind = self.fes[i].vec * x[i][:, None] + np.arange(self.fes[i].vec)[None, :] + self.offset[i]
            inds.append(crt_ind.reshape(-1))

        return np.hstack(inds)

    # (num_cells, num_nodes*vec + ...)
    inds = onp.array(jax.vmap(find_ind)(*self.cells_list))
    self.I = onp.empty(0, dtype=onp.int32)
    self.J = onp.empty(0, dtype=onp.int32)
    self.cells_list_face_list = []

    for i, boundary_inds in enumerate(self.boundary_inds_list):
        cells_list_face = [cells[boundary_inds[:, 0]] for cells in self.cells_list] # [(num_selected_faces, num_nodes), ...]
        inds_face = onp.array(jax.vmap(find_ind)(*cells_list_face)) # (num_selected_faces, num_nodes*vec + ...)
        I_face = onp.repeat(inds_face[:, :, None], inds_face.shape[1], axis=2).reshape(-1)
        J_face = onp.repeat(inds_face[:, None, :], inds_face.shape[1], axis=1).reshape(-1)
        self.I = onp.hstack((self.I, I_face))
        self.J = onp.hstack((self.J, J_face))
        self.cells_list_face_list.append(cells_list_face)

    self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(*self.cells_list) # (num_cells, num_nodes + ...)

    dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
    # TODO: dumb_array_dof is useless?
    dumb_array_node = [np.zeros(fe.num_nodes) for fe in self.fes]
    # _, unflatten_fn_node = jax.flatten_util.ravel_pytree(dumb_array_node)
    _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

    dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
    dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(dumb_sol_list)
    self.num_total_dofs_all_vars = len(dumb_dofs)

    self.num_nodes_cumsum = onp.cumsum([0] + [fe.num_nodes for fe in self.fes])

    self.initialize_geometric_quantities()

    self.internal_vars = ()
    self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
    self.custom_init(*self.additional_info)
    self.pre_jit_fns()
