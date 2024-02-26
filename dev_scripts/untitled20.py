import numpy as np
import pyvista as pv
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt
from mcgs import monte_carlo_grain_structure as mcgs
# ===================================================
#PXGS = mcgs();
#PXGS.simulate()

#######################################################################
# Parameters
xsz = 100;
ysz = 100;
zsz = 100;
Smax = 16;
mcint_of_interest = 49;
zslice = 1;


# Path for the .ctf file
filePathBase = "C:/Development/M2MatMod/UPXO-CPFEM-Preperation/mcgs 175x175x175/TEX1_mcint10_Z_incr1/"
filePathBase = "C:/Development/M2MatMod/UPXO-CPFEM-Preperation/mcgs 20x20x20 S16/Texture instance 1 mslice " + str(mcint_of_interest) + '/'
filePathBase = "C:/Development/M2MatMod/UPXO-CPFEM-Preperation/mcgs 50x50x50 S16/Texture instance 1 mslice " + str(mcint_of_interest) + '/'

phi1 = np.random.rand(Smax)*180
psi = np.random.rand(Smax)*90
phi2 = np.random.rand(Smax)*180

#######################################################################
xgr = PXGS.xgr
ygr = PXGS.ygr
zgr = PXGS.zgr

# Generate a structure that defines connectivity for 3D labeling
s = generate_binary_structure(3, 3).astype('uint8')  # 3D connectivity (for voxels directly touching each other)

filePathBase = "C:/Development/M2MatMod/UPXO-CPFEM-Preperation/mcgs 50x50x50 S16/Texture instance 1 mslice " + str(mcint_of_interest) + '/'


gs = PXGS.gs[mcint_of_interest]
state_matrix = gs.s.astype('uint8');

gid_matrix = np.zeros_like(state_matrix);

gid = 1;
for si in list(range(1, PXGS.uisim.S+1)):
    print('-----------------------------')
    print('Calculating grains and compiling gid_matrix @s=', si)
    # Label each grain in the matrix
    labeled_matrix, num_grains = label(state_matrix==si, structure=s)
    print('Number of grains = ', num_grains)
    if num_grains:
        unique_numbers_nonzero = np.unique(labeled_matrix[labeled_matrix != 0])
        print('Computing grain_pixels dictionary')
        grain_pixels = {number: np.argwhere(labeled_matrix == number) for number in unique_numbers_nonzero}
        print('Finished computing grain_pixels dictionary')
        print('Populating gid_matrix')
        for gpix in grain_pixels.keys():
            LOCS = grain_pixels[gpix]
            for L in LOCS:
                gid_matrix[L[0], L[1], L[2]] = gid
            gid += 1
    else:
        pass



grid = pv.StructuredGrid(xgr, ygr, zgr)
grid['values'] = state_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
grid['gid_values'] = gid_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
# Save the grid to a VTK file
grid.save(filePathBase+'yellowSubmarine2b.vtk')



header = """Channel Text File
Prj\tUPXOGeneratedMicrostructure
Author\tDr. Sunil
JobMode\tGrid
XCells\t{}
YCells\t{}
XStep\t1.0
YStep\t1.0
AcqE1\t0
AcqE2\t0
AcqE3\t0
Euler angles refer to Sample Coordinate system (CS0)!	Mag	0.0000	Coverage	0	Device	0	KV	0.0000	TiltAngle	0.0000	TiltAxis	0	DetectorOrientationE1	0.0000	DetectorOrientationE2	0.0000	DetectorOrientationE3	0.0000	WorkingDistance	0.0000	InsertionDistance	0.0000
Phases\t1
3.614;3.614;3.614	90.000;90.000;90.000	Copper	11	0			Created from UPXO
Phase	X	Y	Bands	Error	Euler1	Euler2	Euler3	MAD	BC	BS""".format(xsz, ysz)
zslice = 1;
state_matrix_slice = state_matrix[:, :, zslice]
xgr_slice = xgr[:, :, zslice]
ygr_slice = ygr[:, :, zslice]
phase_name = 1;
# Assume random Bunge's Euler angles for each pixel (in degrees)
# Header information for the .ctf file (you might need to adjust these values)

data_format = "{phase}\t{x:.3f}\t{y:.3f}\t{bands}\t{error}\t{euler1:.5f}\t{euler2:.5f}\t{euler3:.5f}\t{mad:.5f}\t{bc}\t{bs}"

for zslice in range(zsz):
    print('Writinf .ctf file for slice no: ', zslice, '/', zsz)
    state_matrix_slice = state_matrix[:, :, zslice]
    xgr_slice = xgr[:, :, zslice]
    ygr_slice = ygr[:, :, zslice]
    phase_name = 1;
    # Assume random Bunge's Euler angles for each pixel (in degrees)
    # Header information for the .ctf file (you might need to adjust these values)
    # Preparing the data to write to the file
    data_lines = [header]
    for r in range(ysz):
        for c in range(xsz):
            # Simulating Bands, Error, and MAD values
            bands = np.random.randint(1, 10)
            error = np.random.randint(0, 2)
            mad = np.random.rand()
            current_state = state_matrix_slice[r][c]
            # BC and BS are randomly generated within specified ranges
            bc, bs = np.random.randint(50, 150), np.random.randint(1, 10)

            # Preparing data line
            data_line = data_format.format(
                # phase=state_matrix_slice[r][c],
                phase=phase_name,
                x=xgr_slice[r][c],
                y=ygr_slice[r][c],
                bands=bands,
                error=error,
                euler1=phi1[current_state-1],
                euler2=psi[current_state-1],
                euler3=phi2[current_state-1],
                mad=mad,
                bc=bc,
                bs=bs
            )
            data_lines.append(data_line)

    # Joining all data lines into a single string to simulate the file content
    ctf_data = "\n".join(data_lines)


    #filePathBase = "C:/Development/M2MatMod/mtex/"
    file_path = filePathBase + "slice_"+str(zslice)+".ctf"

    # Writing the data to a .ctf file
    with open(file_path, 'w') as file:
        file.write(ctf_data)

ebsdFilePath = "C:/Development/M2MatMod/mtex/slice_m40_z0"
ebsdFilePath = "D:/EBSD Datasets/OFHC_Cu_dogbone_MAPEXTRACTS_DRF3/Nx37_Ny4_subdomains_LATh.36/Nx1_Ny1"
ebsdFilePath = "D:/EBSD Datasets/CuCrZr/dataset1/map"
ebsdMap = ebsd.Map(ebsdFilePath, dataType='OxfordText')

ebsdMap.xDim
ebsdMap.yDim
pixel_area = ebsdMap.stepSize**2

dir(ebsdMap)

ebsdMap.buildQuatArray()
#ebsdMap.loadSlipSystems("cubic_fcc")
#ebsdMap.plotIPFMap([0,0,1],plotScaleBar=True,plotGBs=False)
ebsdMap.findBoundaries(boundDef = 6)
ebsdMap.findGrains(minGrainSize=10)
ebsdMap.calcGrainMisOri(calcAxis = True)

ebsdMap.grains[1][0]

plt.imshow(ebsdMap.grains <= 0)

ngrains = len(ebsdMap.grains)

ebsdMap.plotEulerMap()
ebsdMap.plotGrainMap()
ebsdMap.plotPhaseMap()
ebsdMap.plotBoundaryMap()
ebsdMap.plotGrainNumbers()
ebsdMap[0].plotMisOri()
ebsdMap[0].grainOutline()
# METHODS ON INDIVIDUAL GRAINS
ebsdMap[0].plotOutline()
