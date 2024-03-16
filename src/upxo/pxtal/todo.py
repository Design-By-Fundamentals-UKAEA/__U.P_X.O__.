"""
LHF -- Low Hanging Fruit
FOL -- Make this First On the List
================================================
TOP PRIORITY in 2D grain structures
* FOL. Map texture component orientation mapping for 2D
* LHF. Remove single pixel grains by merging
* Remove straight line grains by merging
* LHF. State export to .dat files and .bin files
* LHF. Make finer
* LHF. Make coarser
* LHF. Make MCGS from EBSD grain structures from DefDAP
================================================
TOP PRIORITY in 3D grain structures
* FOL. LHF. Re-structure mcgs3_temporal_slice.py to suyit 3D grain structueres
* FOL. LHF. CTF exporter
* FOL. LHF. Import graoin structure from .VTK file.
* FOL. LHF. [UPXO --> .CTF files --> DREAM 3D --> merge grains --> .VTK -->
             UPXO --> Extract GB --> Extract GB surfaces --> Develop surfaces -->
             Make volume from closing surfaces --> Make tet mesh --> Export mesh
             to .inp]
* FOL. LHF. Calculate grain indices
* FOL. LHF. Generate gid-s and s:gid maps
* FOL. LHF. Calculate grain boundary surfaces
* FOL. LHF. Calculate grain volumes
* FOL. LHF. Calculate grainboundary surface pixellated area
* FOL. LHF. Calculate grain level bounding box: use scipy.ndimage.find_objects
* Export to .inp file, natively for hexahedral elements
* LHF. Try open3D for simple grain structure labelling, property estimation,
grain boundary surface extraction
================================================
LOW PRIORITY in 2D grain structures
* C program for MOnte-Carlo simulationcore loops
* Export to .inp file, natively for quad elements with midside nodes
* Export to .inp file, natively for tri elements type 1
* Export to .inp file, natively for tri elements type 2
* Export to .inp file, natively for tri elements type 1 with midside nodes
* Export to .inp file, natively for tri elements type 2 with midside nodes
* Generate scikit.oimage.blobs on grain boundaries, associate a unique ID, mcstate and XORI.
================================================
LOW PRIORITY in 3D grain structures
* C program for MOnte-Carlo simulationcore loops
* Export to .inp file, natively for hex elements with midside nodes
* Export to .inp file, natively for tet elements type 1
* Export to .inp file, natively for tet elements type 2
* Export to .inp file, natively for tet elements type 1 with midside nodes
* Export to .inp file, natively for tet elements type 2 with midside nodes
* Generate scikit.oimage.blobs on grain boundaries, associate a unique ID, mcstate and XORI.
================================================
================================================
================================================
"""
