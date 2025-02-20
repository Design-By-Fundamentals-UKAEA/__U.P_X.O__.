    def _merge_two_grains_(self, parent_gid, other_gid, print_msg=False):
        """Low level merge operartion. No checks done. Just merging.

        Parameters
        ----------
        parent_gid: int
            Parent grain ID number.
        other_gid: int
            Otrher grain ID number.
        print_msg: bool
            Defgaults to False.

        Returns
        -------
        None

        Usage
        -----
        Internal use only.
        """
        self.lgi[self.lgi == other_gid] = parent_gid
        if print_msg:
            print(f"Grain {other_gid} merged with grain {parent_gid}.")

    def merge_two_neigh_grains(self, parent_gid, other_gid,
                               check_for_neigh=True, simple_merge=True):
        """
        Merge other_gid grain to the parent_gid grain.

        Paramters
        ---------
        parent_gid:
            Grain ID of the parent.
        other_gid:
            Grain ID of the other grain being merged into the parent.
        check_for_neigh: bool.
            If True, other_gid will be checked if it can be merged to the
            parent grain. Defaults to True.

        Returns
        -------
        merge_success: bool
            True, if successfully merged, else False.
        """
        def MergeGrains():
            if simple_merge:
                self._merge_two_grains_(parent_gid, other_gid, print_msg=False)
                merge_success = True
            else:
                print("Special merge process. To be developed.")
                merge_success = False  # As of now, this willd efault to False.
            return merge_success
        # ---------------------------------------
        if not check_for_neigh:
            merge_success = MergeGrains()
        else:
            if check_for_neigh and not self.check_for_neigh(parent_gid, other_gid):
                # print('Check for neigh failed. Nothing merged.')
                merge_success = False
            # ---------------------------------------
            if any((check_for_neigh, self.check_for_neigh(parent_gid, other_gid))):
                merge_success = MergeGrains()
                # print(f"Grain {other_gid} merged with grain {parent_gid}.")
        return merge_success

    def perform_post_grain_merge_ops(self, merge_success, merged_gid):
        self.renumber_gid_post_merge(merged_gid)
        self.recalculate_ngrains_post_merge()
        # Update lgi
        # Update neigh_gid
        pass

    def renumber_gid_post_grain_merge(self, merged_gid):
        # self._gid_bf_merger_ = deepcopy(self.gid) # May nor be needed
        GID_left = self.gid[0:merged_gid-1]
        GID_right = [gid-1 for gid in self.gid[merged_gid:]]
        self.gid = GID_left + GID_right

    def recalculate_ngrains_post_grain_merge(self):
        # gid must have been recalculated for tjhis as a pre-requisite.
        self.n = len(self.gid)

    def renumber_lgi_post_grain_merge(self, merged_gid):
        LGI_left = self.lgi[self.lgi < merged_gid]
        self.lgi[self.lgi > merged_gid] -= 1
