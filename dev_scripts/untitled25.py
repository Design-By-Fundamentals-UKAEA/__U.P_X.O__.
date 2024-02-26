import os
import numpy as np
import pyvista as pv
from scipy.ndimage import label, generate_binary_structure
import matplotlib.pyplot as plt
import math

xsz = math.floor((PXGS.uigrid.xmax-PXGS.uigrid.xmin)/PXGS.uigrid.xinc);
ysz = math.floor((PXGS.uigrid.ymax-PXGS.uigrid.ymin)/PXGS.uigrid.yinc);
zsz = math.floor((PXGS.uigrid.zmax-PXGS.uigrid.zmin)/PXGS.uigrid.zinc);
Smax = PXGS.uisim.S;

slice_incr = 1;

slices = list(range(0, zsz, slice_incr))

phase_name = 1;

phi1 = np.random.rand(Smax)*180
psi = np.random.rand(Smax)*90
phi2 = np.random.rand(Smax)*180
textureInstanceNumber = 1;

xgr = PXGS.xgr
ygr = PXGS.ygr
zgr = PXGS.zgr

s = generate_binary_structure(3, 3).astype('uint8')  # 3D connectivity

mcints_of_interest = [49, 5, 10, 15, 20, 30]
rootFolder = "C:/Development/M2MatMod/UPXO-CPFEM-Preperation/mcgs " + str(xsz)+"x"+str(ysz)+"x"+str(zsz) + " S"+str(PXGS.uisim.S)
try:
    os.mkdir(rootFolder)
except FileExistsError:
    print(f"Directory exists: {filePathBase}")
# ===============================================================
header = """Channel Text File
Prj\tUPXO_Synthetic_Grain_Structure
Author\tDr. Sunil Anandatheertha
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
# ===============================================================
NG_m = {i: 0 for i in mcints_of_interest}

for mcint_of_interest in mcints_of_interest:
    filenameSuffix = "texture instance "+str(textureInstanceNumber)+" mctime step " + str(mcint_of_interest)
    filePathBase = rootFolder + "/" + filenameSuffix + '/'
    try:
        os.mkdir(filePathBase)
    except FileExistsError:
        print(f"Directory exists: {filePathBase}")
    # -------------------------------------
    gs = PXGS.gs[mcint_of_interest]
    state_matrix = gs.s.astype('uint8');
    gid_matrix = np.zeros_like(state_matrix);
    # -------------------------------------
    # Wroite VTK file now
    grid = pv.StructuredGrid(xgr, ygr, zgr)
    grid['values'] = state_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
    grid['gid_values'] = gid_matrix.flatten(order='F')  # Flatten in Fortran order to match the VTK's indexing
    # Save the grid to a VTK file
    grid.save(filePathBase+'/'+filenameSuffix+'.vtk')
    # -------------------------------------
    gid = 1;
    ngs = [0 for i in range(PXGS.uisim.S)]
    print(40*'-')
    print('Calculating grains and compiling gid_matrix @temporal slice =', mcint_of_interest)
    for si in list(range(1, PXGS.uisim.S+1)):
        labeled_matrix, num_grains = label(state_matrix==si, structure=s)
        print(f'mcstate {si}: ngrains = {num_grains}. Finishing.')
        ngs[si-1] = num_grains
        if num_grains:
            unique_numbers_nonzero = np.unique(labeled_matrix[labeled_matrix != 0])
            # Compute grain_pixels dictionary
            grain_pixels = {number: np.argwhere(labeled_matrix == number) for number in unique_numbers_nonzero}
            # Populate gid_matrix
            for LOCS in grain_pixels.values():
                # Loop through number of grains
                for L in LOCS:
                    # Loop through pixels belongimg to each grain
                    gid_matrix[L[0], L[1], L[2]] = gid
                gid += 1
        else:
            pass
    # -------------------------------------
    NG_m[mcint_of_interest] = np.sum(ngs)
    # -------------------------------------
    # Header information for the .ctf file
    data_format = "{phase}\t{x:.3f}\t{y:.3f}\t{bands}\t{error}\t{euler1:.5f}\t{euler2:.5f}\t{euler3:.5f}\t{mad:.5f}\t{bc}\t{bs}"
    # -------------------------------------
    # Take slices and export the slice as .ctf file
    for zslice in slices:
        print('Writinf .ctf file for slice no: ', zslice, '/', zsz)
        state_matrix_slice = state_matrix[:, :, zslice]
        xgr_slice = xgr[:, :, zslice]
        ygr_slice = ygr[:, :, zslice]
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
