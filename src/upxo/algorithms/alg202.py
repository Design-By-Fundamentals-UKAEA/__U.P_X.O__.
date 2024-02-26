def mc_iterations_2d_alg202(self):
    """


    Returns
    -------
    None.

    """

    # Begin modified Markov-Chain annealing iterations
    fully_annealed = False
    for m in range(self.uisim.mcsteps):
        if self.S.min() == self.S.max():
            print(f'Single crystal achieved at iteration {m}.')
            print('Stopping the simulation.')
            fully_annealed = True
        else:
            for s0 in list(range(self.S.shape[0])):  # along axis 0
                s00, s01, s02 = s0+0, s0+1, s0+2
                for s1 in list(range(self.S.shape[1])):  # along axis 1
                    s10, s11, s12 = s1+0, s1+1, s1+2
                    ssub_00 = self.S[self.AIA0[s00, s10],
                                        self.AIA1[s00, s10]]
                    ssub_01 = self.S[self.AIA0[s01, s10],
                                        self.AIA1[s01, s10]]
                    ssub_02 = self.S[self.AIA0[s02, s10],
                                        self.AIA1[s02, s10]]
                    ssub_10 = self.S[self.AIA0[s00, s11],
                                        self.AIA1[s00, s11]]
                    ssub_11 = self.S[self.AIA0[s01, s11],
                                        self.AIA1[s01, s11]]
                    ssub_12 = self.S[self.AIA0[s02, s11],
                                        self.AIA1[s02, s11]]
                    ssub_20 = self.S[self.AIA0[s00, s12],
                                        self.AIA1[s00, s12]]
                    ssub_21 = self.S[self.AIA0[s01, s12],
                                        self.AIA1[s01, s12]]
                    ssub_22 = self.S[self.AIA0[s02, s12],
                                        self.AIA1[s02, s12]]
                    Neigh = [ssub_00, ssub_01, ssub_02,
                                ssub_10, ssub_11, ssub_12,
                                ssub_20, ssub_21, ssub_22]
                    if min(Neigh) != max(Neigh):
                        DelH1 = int(ssub_11 == ssub_00) + \
                                int(ssub_11 == ssub_01) + \
                                int(ssub_11 == ssub_02) + \
                                int(ssub_11 == ssub_10) + \
                                int(ssub_11 == ssub_12) + \
                                int(ssub_11 == ssub_20) + \
                                int(ssub_11 == ssub_21) + \
                                int(ssub_11 == ssub_22)
                        # ---------------------------------------------
                        # If the sampling is to be selected without
                        # weightage to dominant neighbour state, then:
                        Neigh = set([x for x in Neigh if x != ssub_11])
                        # If the sampling is to be selected with
                        # weightage to dominant neighbour state, then:
                        # Neigh = [x for x in Neigh if x != ssub_11]
                        # ---------------------------------------------
                        ssub_11_b = sample_rand(Neigh, 1)[0]
                        DelH2 = int(ssub_11_b == ssub_00) + \
                                int(ssub_11_b == ssub_01) + \
                                int(ssub_11_b == ssub_02) + \
                                int(ssub_11_b == ssub_10) + \
                                int(ssub_11_b == ssub_12) + \
                                int(ssub_11_b == ssub_20) + \
                                int(ssub_11_b == ssub_21) + \
                                int(ssub_11_b == ssub_22)
                        if DelH2 >= DelH1:
                            self.S[s0, s1] = ssub_11_b
                        elif self.uisim.consider_boltzmann_probability:
                            if self.uisim.s_boltz_prob[int(ssub_11_b-1)] < rand.random():
                                self.S[s0, s1] = ssub_11_b
        if self.display_messages:
            if m % self.uiint.mcint_promt_display == 0:
                print("Temporal step no.", m)
        cond_1 = m % self.uiint.mcint_save_at_mcstep_interval == 0.0
        if cond_1 or fully_annealed:
            # self.tslices.append(m)
            # Create and add grain structure data structure template
            self.add_gs_data_structure_template(m=m,
                                                dim=self.uigrid.dim)
            self.gs[m].s = deepcopy(self.S)
            if self.display_messages:
                print(f'State updated @ mc step {m}')
            print('__________________________')