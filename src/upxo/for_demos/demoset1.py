import matplotlib.pyplot as plt
from upxo.geoEntities.mulpoint2d import mulpoint2d
from upxo.initialize import gsets


class DEMO_MCGS_2D():
    """
    Generate a 2D temporal slices of grain structure generated using
    modified Q-state Pott's model.

    NOTE1: Make sure you have updated the excel input dashboard file.
    NOTE2: Use drop down wherever available in the excel file.
    NOTE3: Keep to the following values in the excel file:
        dim=2, xmax=ymax=25, xinc=yinc=1, S=32, mcsteps=10,
        mcalg=202, mcint_save_at_mcstep_interval=2

    SCRIPT TO USE THIS DEMONSTRATION
    --------------------------------

    demo = DEMO_MCGS_2D()
    demo.basic_mcgs2d()  # Make the grain structure
    demo.see_available_gs()  # View all availavbel grain structres
    demo.set_toi(6)  # Set the tslice of interest

    demo.plot_gs()  # View the grain structure
    areas = demo.get_grain_areas()  # Get areas of all grains
    demo.mcgs2d_to_vtgs2d()  # Generate VTGS equivalent of mcgs
    demo.plot_largest_grain()
    demo.get_neighbouring_grains(1)
    demo.plot_neigh_grains_2d(2)
    demo.plot_grains([1, 3, 5, 7, 20])
    """

    def __init__(self):
        self.gsdb = None
        self.toi = None

    def basic_mcgs2d(self):
        """Generate basic Grain Structure Data Base"""
        self.gsdb = gsets()
        self.gsdb.initialize(gstype='mc2d')

    def see_available_gs(self):
        """See the available grain structures"""
        # Note: NOt to worry about 'l0'. Just go with it for now.
        return self.gsdb.db['l0'].tslices

    def set_toi(self, toi):
        """ Set temporal slice of interest """
        self.toi = toi

    def plot_gs(self):
        """ PLot gran structure"""
        self.gsdb.db['l0'].gs[self.toi].plot()

    def get_grain_areas(self):
        """Access the areas of the grain structure. Col 1 is grain id
        and Col 2 is the pixle computed area"""
        areas = self.gsdb.db['l0'].gs[self.toi].prop.area
        return areas

    def mcgs2d_to_vtgs2d(self, see_vtgs=True):
        """COnvert mcgs2d to VTGS"""
        self.gsdb.db['l0'].gs[self.toi].vtgs2d(visualize=see_vtgs)

    def plot_grain(self, gid):
        """Plot a grain by its id"""
        self.gsdb.db['l0'].gs[self.toi].g[gid+1]['grain'].plot()

    def plot_grains(self, gids):
        """
        Plot a grain by its id.
        """
        self.gsdb.db['l0'].gs[self.toi].plot_grains(gids)

    def plot_largest_grain(self):
        """ Plot the largest available grain in the set toi"""
        # To acoid plotting single pixel grains, let's isolate the
        # largest grain
        gid = self.get_grain_areas().idxmax()  # Pandas series
        self.plot_grain(gid)

    def get_neighbouring_grains(self, gid):
        """ Get the ids of grain neighbouring the gid grain"""
        return self.gsdb.db['l0'].gs[self.toi].g[gid]['grain'].neigh

    def plot_neigh_grains_2d(self, gid):
        ngids = self.gsdb.db['l0'].gs[self.toi].g[gid]['grain'].neigh
        self.gsdb.db['l0'].gs[self.toi].plot_grains_gids(ngids,
                                                         gclr='color',
                                                         title="user grains",
                                                         cmap_name='viridis')

    def write_ctf(self, gid):
        self.gsdb.db['l0'].gs[self.toi].ebsd_write_ctf(folder='upxo_ctf',
                                                       file='upxo_mcslice.ctf')

    def write_abaqus_mesh(self):
        femesh = self.gsdb.db['l0'].gs[self.toi].femesh(saa=True, throw=True)
        femesh.export_abaqus_inp_file(folder="DATABASE_FEMESH",
                                      file="UPXO_ABQ_MESH.inp")


def gradient_gs_vtgs_2d():
    """
    Command to test this demo
    -------------------------
    seedmp1 = gradient_gs_vtgs_2d()
    """
    seedmp1 = mulpoint2d(method='random',
                         gridding_technique='pds',
                         sampling_technique='bridson1',
                         xbound=[0, 1], ybound=[0, 1],
                         char_length=[0.15, 0],
                         bridson_sampling_k=10,
                         make_point_objects=True,
                         mulpoint_type='seed',
                         lean='ignore')
    seedmp2 = mulpoint2d(method='random',
                         gridding_technique='pds',
                         sampling_technique='bridson1',
                         xbound=[1, 2], ybound=[0, 1],
                         char_length=[0.05, 0],
                         bridson_sampling_k=10,
                         make_point_objects=True,
                         mulpoint_type='seed',
                         lean='ignore')
    seedmp3 = mulpoint2d(method='random',
                         gridding_technique='pds',
                         sampling_technique='bridson1',
                         xbound=[1, 2], ybound=[1, 2],
                         char_length=[0.2, 0],
                         bridson_sampling_k=10,
                         make_point_objects=True,
                         mulpoint_type='seed',
                         lean='ignore')
    mps = [seedmp2, seedmp3]
    seedmp1.add(toadd='multi_points',
                multi_point_objects_list=mps)

    plt.plot(seedmp1.locx, seedmp1.locy, 'ks')

    return seedmp1
