# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:01:58 2022

@author: chardie
"""
import numpy as np
import os.path
import traceback
import pickle
import math
import numpy as np
import copy
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from scipy import stats

class Project:
    def __init__(self,phi1=None,theta=None,phi2=None,BC=None,phase=None,grains=None,x_map=None,y_map=None,boundaries=None,xc=None,yc=None,scale=None):
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        def_name = text[:text.find('=')].strip()
        self.name = def_name
        self.phi1=phi1
        self.theta=theta
        self.phi2=phi2
        self.BC=BC
        self.phase=phase
        self.grains=grains
        self.x_map=x_map
        self.y_map=y_map
        self.boundaries=boundaries
        self.xc=xc
        self.yc=yc
        self.scale=scale

    @classmethod
    def loadDefDapData(cls, data_file):
        exec('from '+data_file+' import loadDD')
        exec('cls.HRDIC, cls.EBSD, cls.n = loadDD()')

    @staticmethod
    def loadProjectData(file_name):
        with open(file_name+'.pickle', 'rb') as file2:
                project = pickle.load(file2)
        return project

    def extractData(self):
        # Transform Deformed EBSD corrds to DIC undeformed reference frame
        self.phi1=self.HRDIC[0].warpToDicFrame(self.EBSD.eulerAngleArray[0], cropImage=True, order=0, preserve_range=True)
        self.theta=self.HRDIC[0].warpToDicFrame(self.EBSD.eulerAngleArray[1], cropImage=True, order=0, preserve_range=True)
        self.phi2=self.HRDIC[0].warpToDicFrame(self.EBSD.eulerAngleArray[2], cropImage=True, order=0, preserve_range=True)
        self.BC=self.HRDIC[0].warpToDicFrame(self.EBSD.bandContrastArray, cropImage=True, order=0, preserve_range=False)
        self.phase=self.HRDIC[0].warpToDicFrame(self.EBSD.phaseArray, cropImage=True, order=0, preserve_range=False)
        self.grains=self.HRDIC[0].warpToDicFrame(self.EBSD.grains, cropImage=True, order=0, preserve_range=True)
        self.x_map=self.HRDIC[self.n-1].crop(self.HRDIC[self.n-1].x_map)*self.HRDIC[self.n-1].scale/self.HRDIC[self.n-1].binning # in microns. /binning is required because the original pixels were 1/binning in sizes
        self.y_map=self.HRDIC[self.n-1].crop(self.HRDIC[self.n-1].y_map)*self.HRDIC[self.n-1].scale/self.HRDIC[self.n-1].binning # in microns
        self.boundaries=self.HRDIC[0].boundaries
        self.xc=np.linspace(0,np.shape(self.x_map)[1]*self.HRDIC[self.n-1].scale,np.shape(self.x_map)[1])
        self.yc=np.linspace(0,np.shape(self.x_map)[0]*self.HRDIC[self.n-1].scale,np.shape(self.x_map)[0])
        self.scale=self.HRDIC[self.n-1].scale

    def saveProjectData(self, file_name):
        # Save the relevant project data for treatment on a region by region basis
        # Record current directory
        current_directory=os.getcwd()
        # Go to save directory
        #os.chdir(directory)
        #Save class as pickle with name 'self'
        with open(file_name+'.pickle', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        # return to original directory
        os.chdir(current_directory)

class Region:
    def __init__(self):
        pass

#Save region data
    def saveRegionData(self, file_name):
        # Save the relevant project data for treatment on a region by region basis
        # Record current directory
        current_directory=os.getcwd()
        # Go to save directory
        #os.chdir(directory)
        #Save class as pickle with name 'self'
        with open(file_name+'.pickle', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        # return to original directory
        os.chdir(current_directory)

# Load Region Data
    @staticmethod
    def loadRegionData(file_name):
        with open(file_name+'.pickle', 'rb') as file2:
                project = pickle.load(file2)
        return project


#Develop Abaqus Model
#Plot stuff

    # Make region data
    def defineRegion(self,Project,x1, x2, y1, y2):

        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.gsb=40 #Grain sketch border

        hcp=False
        EBSDPostDef=False
        if hcp:
            d_reference_angle=30
        else:
            d_reference_angle=0

       # Convert to degress

        self.phi1=-Project.phi1*180/np.pi
        self.theta=-Project.theta*180/np.pi
        self.phi2=-Project.phi2*180/np.pi-d_reference_angle
        # a final 30 degree rotation is required to transform the reference config from the EBSD to that of ABAQUS

        # convert BC to rounded percent - only required for plotting
        self.BC=np.round(Project.BC*100)

        # Crop all - this is done after importing all data so that the stitching of the DIC and EBSD maps
        # with homologous points is successful. The crop includes an extra border of gsb pixels wide to produce enough grain boundary
        # paths to reach the edge of the domain. It this subsequently cropped off when sketching in Abaqus. Care is needed if the pixel size
        # changes since this will need to be incorporated in the python script which does the sketching.

        x_map=Project.x_map[y1:y2,x1:x2]
        y_map=Project.y_map[y1:y2,x1:x2]
        self.boundaries=Project.boundaries[y1:y2,x1:x2]
        self.grain_map=Project.grains[y1-self.gsb:y2+self.gsb,x1-self.gsb:x2+self.gsb] # add an extra 10 pixels around the map for the abaqus sketch and crop later
        self.grains=Project.grains[y1:y2,x1:x2]

        #Cropping bounds
        self.bounds=[Project.xc[x1],Project.xc[x2-1],Project.yc[y1],Project.yc[y2-1]]# upper indicies include minus 1 due to slicing convention when using : in python

        self.phi1=Project.phi1[y1:y2,x1:x2]
        self.theta=Project.theta[y1:y2,x1:x2]
        self.phi2=Project.phi2[y1:y2,x1:x2]
        self.BC=Project.BC[y1:y2,x1:x2]
        self.phase=Project.phase[y1:y2,x1:x2]

        self.xc_extended=Project.xc[self.x1-self.gsb:self.x2+self.gsb]
        self.yc_extended=Project.yc[self.y1-self.gsb:self.y2+self.gsb]

        self.xc=Project.xc[x1:x2]
        self.yc=Project.yc[y1:y2]

        # Apply corrections for translation and rotation:

        #Rigid body translation:
        x_map_c=x_map-x_map[0,0]
        y_map_c=y_map-y_map[0,0]

        u_prime=0.0
        v_prime=0.0

        if EBSDPostDef:
            u1=x_map_c[0,0] #bottom left
            u2=x_map_c[0,-1] #bottom right
            u3=x_map_c[-1,-1] #top right
            u4=x_map_c[-1,0] #top left

            v1=y_map_c[0,0] #bottom left
            v2=y_map_c[0,-1] #bottom right
            v3=y_map_c[-1,-1] #top right
            v4=y_map_c[-1,0] #top left

            # Define the undeformed and deformed vectors that go from the bottom left corner to the bottom right:
            X=np.array([self.xc[-1],self.yc[0]]) # vector before displacement
            x=np.array([self.xc[-1]+u2,self.yc[0]+v2]) # vector after displacement
            # Make unit vectors
            X=X/np.linalg.norm(X)
            x=x/np.linalg.norm(x)
            # Calculate the angle between the vectors
            dot_prod=np.dot(X,x)
            angle=-np.arccos(dot_prod)
            #Rotate displacements based on position in map...
            #create 2 arrays describing the rotation wrt. each coordinate position
            u_prime=np.zeros(np.shape(x_map_c))
            v_prime=np.zeros(np.shape(x_map_c))

            for i in range(len(self.xc)):
                for j in range(len(self.yc)):
                    u_prime[j,i]=np.cos(angle)*(self.xc[i])-np.sin(angle)*(self.yc[j])-self.xc[i] # rotated vectors with the reference coordinates subtracted to provide u and v of rotation only.
                    v_prime[j,i]=np.sin(angle)*(self.xc[i])+np.cos(angle)*(self.yc[j])-self.yc[j]

        # Correct displacement maps
        self.x_map=x_map_c+u_prime
        self.y_map=y_map_c+v_prime

        #Produce BC data:
        self.LU=x_map_c[:,0]
        self.LV=y_map_c[:,0]

        self.RU=x_map_c[:,-1]
        self.RV=y_map_c[:,-1]

        self.TU=x_map_c[-1,:]
        self.TV=y_map_c[-1,:]

        self.BU=x_map_c[0,:]
        self.BV=y_map_c[0,:]

        # Grain orientation data
        # Meshgrids for x and y coordinates
        x_coord, y_coord=np.meshgrid(self.xc,self.yc)

        # Find center coordinates for each grain
        grain_num=np.unique(self.grains)
        grain_num=grain_num[grain_num>0]

        self.grain_x=np.zeros(len(grain_num))
        self.grain_y=np.zeros(len(grain_num))
        for i in range(len(grain_num)):
            self.grain_x[i]=np.mean(x_coord[self.grains==grain_num[i]])
            self.grain_y[i]=np.mean(y_coord[self.grains==grain_num[i]])

        # Find indicies for the central points

        x_ind=np.zeros(len(self.grain_x))
        y_ind=np.zeros(len(self.grain_x))
        self.grain_num=np.zeros(len(self.grain_x))
        for i in range(len(self.grain_x)):
            x_ind[i]=(np.abs(self.xc - self.grain_x[i])).argmin()
            y_ind[i]=(np.abs(self.yc - self.grain_y[i])).argmin()
            self.grain_num[i]=self.grains[int(x_ind[i]),int(y_ind[i])]

        # List rotation matricies for the various grains

        #define rotation matrices
        # def Rz(angle):
        #     Rz=[[math.cos(angle), math.sin(angle), 0],[-math.sin(angle), math.cos(angle),0],[0,0,1]]
        #     return Rz

        # def Rx(angle):
        #     Rx=[[1,0,0],[0,math.cos(angle),math.sin(angle)],[0,-math.sin(angle), math.cos(angle)]]
        #     return Rx

        # This currently takes the 3 Euler angles at the central coordinate point of each grain
        self.R=np.zeros((len(x_ind),9))
        self.p1=np.zeros(len(x_ind))
        self.t=np.zeros(len(x_ind))
        self.p2=np.zeros(len(x_ind))
        for i in range(len(x_ind)):
            self.p1[i]=self.phi1[int(y_ind[i]),int(x_ind[i])] # math.radians(self.phi1[int(y_ind[i]),int(x_ind[i])])
            self.t[i]=self.theta[int(y_ind[i]),int(x_ind[i])] # math.radians(self.theta[int(y_ind[i]),int(x_ind[i])])
            self.p2[i]=self.phi2[int(y_ind[i]),int(x_ind[i])] # math.radians(self.phi2[int(y_ind[i]),int(x_ind[i])])


            # T=np.dot(Rx(self.t[i]),Rz(self.p1[i]))
            # T=np.dot(Rz(self.p2[i]),T)
            # ROT=np.linalg.inv(T)
            # self.R[i,:]=np.reshape(ROT, 9)

    # Save microstructure morphology for sketching
    def saveSketchData(self):

        grain_map=self.grain_map
        xc=self.xc_extended
        yc=self.yc_extended
        #gsb=self.gsb

        # Kill small grains
        minGrain=False
        if minGrain:
            for i in np.unique(grain_map):
                if len(grain_map[grain_map==i])<100:
                    grain_map[grain_map==i]=0

        #Remove negative numbers
        grain_map[grain_map<0]=0

        rows, columns=np.shape(grain_map)

        while 0 in np.unique(grain_map):
            for i in np.linspace(0,rows-1,rows):
                for j in np.linspace(0,columns-1,columns):
                    if grain_map[int(i),int(j)]==0:
                        local_grains=grain_map[int(max(i-1,0)):int(min(rows,i+2)),int(max(j-1,0)):int(min(columns,j+2))]
                        print(local_grains[local_grains>0])
                        print('row '+str(i)+' column '+str(j)+' local median= ')
                        print(int(stats.mode(local_grains[local_grains>0])[0]))
                        grain_map[int(i),int(j)]=int(stats.mode(local_grains[local_grains>0])[0])

        # Save data to map grain boundaries for abaqus sketch
        x_triple=np.zeros(np.shape(grain_map))
        for i in np.linspace(1,rows-1,rows-1):
            for j in np.linspace(1,columns-1,columns-1):
                x_triple[int(i),int(j)]=np.size(np.unique(grain_map[int(i):int(i)+2,int(j):int(j)+2]))
                # reduce triple points to a single pixel
                if x_triple[int(i),int(j)]==3:
                    num=sum(sum(k==3 for k in np.reshape(x_triple[max(int(i)-1,0):int(i)+2,max(int(j)-1,0):int(j)+2], (1,9))))
                    if num>1:
                        print("i="+str(i)+" and j="+str(j))
                        x_triple[int(i),int(j)]=2

        x_triple[:,0]=x_triple[:,1]
        x_triple[0,:]=x_triple[1,:]

        image=np.zeros(np.shape(x_triple))
        image[x_triple==2]=1
        image[x_triple==3]=1
        skeleton = skeletonize(image, method='lee')
        skeleton[skeleton==255]=2
        skeleton[x_triple==3]=3

        x_triple=skeleton
        boundaries_map=copy.deepcopy(x_triple)

        record_map=np.zeros(np.shape(x_triple))
        dim=len(x_triple)

        border_map=copy.deepcopy(x_triple)
        border_map[1:dim-1,1:dim-1]=0

        boundaries_map[border_map==2]=3
        junction_map=np.zeros(np.shape(boundaries_map))
        junction_map[boundaries_map==3]=3

        junction_indicies_row,junction_indicies_column=np.where(boundaries_map==3)

        junction_rows=junction_indicies_row # hard list of juction indicies
        junction_columns=junction_indicies_column

        #junction_indicies=[junction_indicies_row,junction_indicies_column] # possibly not needed


        # Walk paths where [i, j] is the current location

        paths={} # set paths dictionary
        path_rows={}
        path_columns={}
        path_number=0 # set path counter
        route=[]
        route_row=[]
        route_column=[]
        dead_end_rows=[]
        dead_end_columns=[]
        while np.size(junction_indicies_row)>0: # there are junctions to explore from
            i=junction_indicies_row[0] # row index of first junction
            j=junction_indicies_column[0] # column index of first junction
            route=np.append(route,[i,j])
            route_row=np.append(route_row,i)
            route_column=np.append(route_column, j)
            junction_distance=0
            boundaries_map[i,j]=0 # zero where standing
            record_map[i,j]=record_map[i,j]+1 # record where we have been on the record map
            #if len(path)==200:
                #boundaries_map[max(i-1,0):i+2,max(j-1,0):j+2]
            test=np.unique(boundaries_map[max(i-1,0):i+2,max(j-1,0):j+2]) # What is nearby?
            while np.sum(test)>0:

                if 2 in test:
                    # there is a path to take
                    masked_map=copy.deepcopy(boundaries_map)
                    masked_map[0:max(i-1,0),:]=0
                    masked_map[i+2::,:]=0
                    masked_map[:,0:max(j-1,0)]=0
                    masked_map[:,j+2::]=0
                    local_row, local_column=np.where(masked_map==2) # locate next step
                    #step forward
                    i=local_row[0]
                    j=local_column[0]
                    route=np.append(route,[i,j])
                    route_row=np.append(route_row,i)
                    route_column=np.append(route_column, j)
                    boundaries_map[i,j]=0 #step and zero where standing
                    record_map[i,j]=record_map[i,j]+1 # record where we have been on the record map
                    junction_distance+=1

                    test=np.unique(junction_map[max(i-1,0):i+2,max(j-1,0):j+2]) # Is there a junction nearby?
                    if 3 in test and junction_distance>3:
                        masked_map=copy.deepcopy(junction_map)
                        masked_map[0:max(i-1,0),:]=0
                        masked_map[i+2::,:]=0
                        masked_map[:,0:max(j-1,0)]=0
                        masked_map[:,j+2::]=0
                        local_row, local_column=np.where(masked_map==3) # locate next step
                        if local_row[0]==junction_indicies_row[0] and local_column[0]==junction_indicies_column[0]:# if the junction found is the one we left from.
                            test=np.unique(boundaries_map[max(i-1,0):i+2,max(j-1,0):j+2]) # What is nearby?
                        else:
                            i=local_row[0]
                            j=local_column[0] #step forward
                            junction_indicies_row=np.append(i,junction_indicies_row)
                            junction_indicies_column=np.append(j, junction_indicies_column) # add to list of junctions
                            boundaries_map[i,j]=0 #step and zero where standing
                            route=np.append(route,[i,j])
                            route_row=np.append(route_row,i)
                            route_column=np.append(route_column, j)
                            record_map[i,j]=record_map[i,j]+1 # record where we have been on the record map
                            test=0
                    else:
                        test=np.unique(boundaries_map[max(i-1,0):i+2,max(j-1,0):j+2]) # What is nearby?
                else:# Nowhere to go
                    boundaries_map[i,j]=0 #step and zero where standing
                    record_map[i,j]=record_map[i,j]+1 # record where we have been on the record map
                    nearest_junction_index=np.where(abs(junction_rows-i)**2+abs(junction_columns-j)**2==min(abs(junction_rows-i)**2+abs(junction_columns-j)**2))
                    i=junction_rows[nearest_junction_index[0][0]]# Add nearest junction to the route
                    j=junction_columns[nearest_junction_index[0][0]]
                    if i==junction_indicies_row[0] and j==junction_indicies_column[0]:# if the junction found is the one we left from.
                            test=0
                    else:
                        route=np.append(route,[i,j])
                        route_row=np.append(route_row,i)
                        route_column=np.append(route_column, j)
                        dead_end_rows=np.append(dead_end_rows,i)
                        dead_end_columns=np.append(dead_end_columns,j)
                        test=0

            # Revise junction locations
            junction_indicies_row=junction_indicies_row[1::]
            junction_indicies_column=junction_indicies_column[1::]

            #Store path

            paths[str(path_number)]=route
            path_rows[str(path_number)]=route_row
            path_columns[str(path_number)]=route_column
            route=[] # reset route
            route_row=[]
            route_column=[]
            path_number+=1
            fig, current_ax = plt.subplots()
            plt.contour(record_map)
            plt.scatter(dead_end_columns,dead_end_rows, color='red')
            plt.show()

        path={}
        path_row={}
        path_column={}
        for x in paths:
            print(x)
            if len(paths[x])>2:
                path[x]=paths[x]
                path_row[x]=path_rows[x]
                path_column[x]=path_columns[x]

        path_keys=[i for i in path.keys()]

        # Tie up loose ends
        end_row_indicies=[]
        end_column_indicies=[]
        for x in [str(y) for y in path_keys]:
            end_row_indicies.append(path[x][0])
            end_row_indicies.append(path[x][-2])
            end_column_indicies.append(path[x][1])
            end_column_indicies.append(path[x][-1])

        row_diff=np.zeros((len(end_row_indicies),len(end_row_indicies)))
        col_diff=np.zeros((len(end_row_indicies),len(end_row_indicies)))

        for i in range(len(end_row_indicies)):
            row_diff[:,i]=end_row_indicies-end_row_indicies[i]
            col_diff[:,i]=end_column_indicies-end_column_indicies[i]

        abs_diff=abs(row_diff)+abs(col_diff)

        loose_ends_id=np.zeros(np.shape(abs_diff))
        loose_ends_id[abs_diff<4]=1 # 4 pixel distance used as a threshold
        loose_ends_id[abs_diff==0]=0
        # remove bottom symmetric data
        loose_ends_id=np.triu(loose_ends_id)

        # Find loose ends
        q,r=np.where(loose_ends_id==1)

        # Join loose ends
        for i in range(len(q)):
            path1_no=int(np.floor((q[i])/2))
            path2_no=int(np.floor((r[i])/2))
            #append coords to path1
            if r[i]%2==0:
                # Even - path2 at start
                coords=path[path_keys[path2_no]][0:2]
                print("Start coords for path"+path_keys[path2_no]+str(coords))
            else:
                # Odd - path2 at end
                coords=path[path_keys[path2_no]][-2::]
                print("End coords for path"+path_keys[path2_no]+str(coords))
            if q[i]%2==0:
                # Even - path1 at start
                print("Start of path no"+path_keys[path1_no]+str(path[path_keys[path1_no]][0:2]))
                path[path_keys[path1_no]]=np.append(coords, path[path_keys[path1_no]])
                path_row[path_keys[path1_no]]=np.append(coords[0], path_row[path_keys[path1_no]])
                path_column[path_keys[path1_no]]=np.append(coords[1], path_column[path_keys[path1_no]])

            else:
                # Odd - path 1 at end
                print("End of path no"+path_keys[path1_no]+str(path[path_keys[path1_no]][-2::]))
                path[path_keys[path1_no]]=np.append(path[path_keys[path1_no]], coords)
                path_row[path_keys[path1_no]]=np.append(path_row[path_keys[path1_no]], coords[0],)
                path_column[path_keys[path1_no]]=np.append(path_column[path_keys[path1_no]], coords[1])


        # Save the data

        current_directory=os.getcwd()

        if os.path.exists(current_directory+"\\GrainBoundaries"):
            pass
        else:
            os.mkdir(current_directory+"\\GrainBoundaries")

        os.chdir(current_directory+"\\GrainBoundaries")

        bounds=self.bounds
        np.save('bounds.npy',bounds)

        scale=max(yc[1]-yc[0],xc[1]-xc[0])
        GBcoord={}
        GBlist=[]
        i=1
        for x in path:
        #    if len(path[x])>2:
                xcoords=path_column[x]*scale # convert to microns
                xcoords=xcoords+np.ones(np.shape(xcoords))*xc[0] # position relative to global map
                xcoords=np.round(xcoords,decimals=1)
                ycoords=path_row[x]*scale # convert to microns
                ycoords=ycoords+np.ones(np.shape(ycoords))*yc[0]
                ycoords=np.round(ycoords,decimals=1)
                if len(path[x])<3:
                    print(x+" is too small to save")
                else:
                    plt.plot(xcoords,ycoords, color='green')
                    plt.annotate(x,(xcoords[int(np.ceil(len(xcoords)/2))],ycoords[int(np.ceil(len(ycoords)/2))]))
                    temparray=np.c_[xcoords, ycoords]
                    GBcoord[str(i)]=temparray.reshape(np.size(temparray))
                    GBlist.append(temparray.reshape(np.size(temparray)).tolist())
                    #np.savetxt('allign-gb-'+str(i)+'.txt',GBcoord[str(i)],fmt='%.18e', newline = "\t")
                    i+=1
        import pickle

        with open('GBlist', 'wb') as pickle_file:
            pickle.dump(GBlist, pickle_file,protocol=2)

        phase=1
        UMAT=np.zeros((len(self.grain_num),8))
        for i in range(len(self.grain_num)):
            UMAT[i]=[self.grain_x[i], self.grain_y[i], self.p1[i],self.t[i],self.p2[i],self.grain_num[i],phase,0]

        with open('UMAT', 'wb') as pickle_file:
            pickle.dump(UMAT, pickle_file, protocol=2)

        os.chdir(current_directory)


        # Save Material Data for each grain
        #order the list of indicies to the grain number order defined in abaqus




    def extractBoundaryDisplacements(self, file_name):

        with open(file_name) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]


        # Get node numbers

        # All nodes

        edgeSets={}

        for y in ['*Nset, nset=LEFT', '*Nset, nset=RIGH','*Nset, nset=TOP,','*Nset, nset=BOT,']:
            nodeslist=''
            i=0
            for x in content:

                if x[0:16]==y:
                    print(x[0:16])
                    print(content[i+1][0])
                    print(i)
                    j=i
                    while content[j+1][0]!='*':
                        nodeslist=nodeslist+','+content[j+1]
                        j+=1

                i+=1

            nodeslist=nodeslist.split(",")
            nodeslist = [i for i in nodeslist if i]
            #nodeslist=nodeslist[1:]
            nodeslist=[int(i) for i in nodeslist]
            edgeSets[y]=nodeslist


        # Get nodal coordinates:
        # i=0
        # coords=[]
        # for x in content:
        #         if x[0:5]=='*Node':
        #             j=0
        #             while content[j+i+1][0]!='*':
        #                 coords=np.append(coords, content[j+i+1])
        #                 j+=1
        #         i+=1
        coords=content[[i[0:5] for i in content].index('*Node')+1:[i[0:8] for i in content].index('*Element')]

        coords=[i.split(",")for i in coords]
        z=0
        for j in coords:
            coords[z]=[float(i) for i in coords[z]]
            coords[z][0]=int(coords[z][0])
            z+=1

        # combine nodes at edges with respective coordinates:

        for y in ['*Nset, nset=LEFT', '*Nset, nset=RIGH','*Nset, nset=TOP,','*Nset, nset=BOT,']:
            x=0
            for j in edgeSets[y]:
                #idx=[i[0] for i in coords].index(int(j)) # index of coord information for each node in edgeset

                edgeSets[y][x]=np.append(edgeSets[y][x], coords[int(j)-1][1:4])
                x+=1

        BCyc=np.c_[self.yc,self.LU,self.LV,self.RU,self.RV]
        BCxc=np.c_[self.xc,self.TU,self.TV,self.BU,self.BV]

        #   Left Edge - for each node locate corresponding two points either side with DIC (in y axis)
        LV=np.zeros(len(edgeSets['*Nset, nset=LEFT']))
        LU=np.zeros(len(edgeSets['*Nset, nset=LEFT']))
        NL=[int(0) for i in range(len(edgeSets['*Nset, nset=LEFT']))]

        x=0
        for i in edgeSets['*Nset, nset=LEFT']:
            norm=BCyc[:,0]-i[2]
            j=0
            while j<len(norm):
                if all(norm<0): # If the model extends beyond the region we have displacement data for
                    NL[x]=int(i[0])
                    LU[x]=BCyc[-1,1]
                    LV[x]=BCyc[-1,2]
                    j=len(norm)     # Done - leave loop
                    print("all found negative")
                    print(norm)
                elif all(norm>0): # If model starts before the region we have displacement data for
                    NL[x]=int(i[0])
                    LU[x]=BCyc[0,1]
                    LV[x]=BCyc[0,2]
                    j=len(norm)     # Done - leave loop
                    print("all found positive")
                    print(norm)
                elif norm[j]<0 and norm[j+1]>0:
                    #print(j)
                    #print(norm[j])
                    #print(norm[j+1])
                    #print(BCyc[j,1])
                    NL[x]=int(i[0])
                    LU[x]=BCyc[j,1]+norm[j]*-1*(BCyc[j+1,1]-BCyc[j,1])/(BCyc[j+1,0]-BCyc[j,0])
                    LV[x]=BCyc[j,2]+norm[j]*-1*(BCyc[j+1,2]-BCyc[j,2])/(BCyc[j+1,0]-BCyc[j,0])
                    j=len(norm)     # Done - leave loop
                j+=1
            x+=1

        fig = plt.figure()
        plt.title('Left Edge U Displacements')
        x=0
        for i in edgeSets['*Nset, nset=LEFT']:
            plt.scatter(i[2],LU[x])
            x+=1
        plt.plot(BCyc[:,0],BCyc[:,1])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('U Displacement (um)')
        plt.show()


        fig = plt.figure()
        plt.title('Left Edge V Displacements')
        x=0
        for i in edgeSets['*Nset, nset=LEFT']:
            plt.scatter(i[2],LV[x])
            x+=1
        plt.plot(BCyc[:,0],BCyc[:,2])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('V Displacement (um)')
        plt.show()

        #   Right Edge - for each node locate corresponding two points either side with DIC (in y axis)
        RV=np.zeros(len(edgeSets['*Nset, nset=RIGH']))
        RU=np.zeros(len(edgeSets['*Nset, nset=RIGH']))
        NR=[int(0) for i in range(len(edgeSets['*Nset, nset=RIGH']))]

        x=0
        for i in edgeSets['*Nset, nset=RIGH']:
            norm=BCyc[:,0]-i[2]
            j=0
            while j<len(norm):
                if all(norm<0): # If the model extends beyond the region we have displacement data for
                    NR[x]=int(i[0])
                    RU[x]=BCyc[-1,3]
                    RV[x]=BCyc[-1,4]
                    j=len(norm)     # Done - leave loop
                    print("all found negative")
                    print(norm)
                elif all(norm>0): # If model starts before the region we have displacement data for
                    NR[x]=int(i[0])
                    RU[x]=BCyc[0,3]
                    RV[x]=BCyc[0,4]
                    j=len(norm)     # Done - leave loop
                    print("all found positive")
                    print(norm)
                elif norm[j]<0 and norm[j+1]>0:
                    #print(j)
                    #print(norm[j])
                    #print(norm[j+1])
                    #print(BCyc[j,1])
                    NR[x]=int(i[0])
                    RU[x]=BCyc[j,3]+norm[j]*-1*(BCyc[j+1,3]-BCyc[j,3])/(BCyc[j+1,0]-BCyc[j,0])
                    RV[x]=BCyc[j,4]+norm[j]*-1*(BCyc[j+1,4]-BCyc[j,4])/(BCyc[j+1,0]-BCyc[j,0])
                    j=len(norm)     # Done - leave loop
                j+=1
            x+=1

        fig = plt.figure()
        plt.title('Right Edge U Displacements')
        x=0
        for i in edgeSets['*Nset, nset=RIGH']:
            plt.scatter(i[2],RU[x])
            x+=1
        plt.plot(BCyc[:,0],BCyc[:,3])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('U Displacement (um)')
        plt.show()


        fig = plt.figure()
        plt.title('Right Edge V Displacements')
        x=0
        for i in edgeSets['*Nset, nset=RIGH']:
            plt.scatter(i[2],RV[x])
            x+=1
        plt.plot(BCyc[:,0],BCyc[:,4])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('V Displacement (um)')
        plt.show()


        #   Top Edge - for each node locate corresponding two points either side with DIC (in y axis)
        TV=np.zeros(len(edgeSets['*Nset, nset=TOP,']))
        TU=np.zeros(len(edgeSets['*Nset, nset=TOP,']))
        NT=[int(0) for i in range(len(edgeSets['*Nset, nset=TOP,']))]

        x=0
        for i in edgeSets['*Nset, nset=TOP,']:
            norm=BCxc[:,0]-i[1]
            j=0
            while j<len(norm)+1:
                if all(norm<0): # If the model extends beyond the region we have displacement data for
                    NT[x]=int(i[0])
                    TU[x]=BCxc[-1,1]
                    TV[x]=BCxc[-1,2]
                    j=len(norm)+1     # Done - leave loop
                    print("all found negative")
                    print(norm)
                elif all(norm>0): # If model starts before the region we have displacement data for
                    NT[x]=int(i[0])
                    TU[x]=BCxc[0,1]
                    TV[x]=BCxc[0,2]
                    j=len(norm)+1     # Done - leave loop
                    print("all found positive")
                    print(norm)
                elif norm[j]<0 and norm[j+1]>0:
                    #print(j)
                    #print(norm[j])
                    #print(norm[j+1])
                    #print(BCyc[j,1])
                    NT[x]=int(i[0])
                    TU[x]=BCxc[j,1]+norm[j]*-1*(BCxc[j+1,1]-BCxc[j,1])/(BCxc[j+1,0]-BCxc[j,0])
                    TV[x]=BCxc[j,2]+norm[j]*-1*(BCxc[j+1,2]-BCxc[j,2])/(BCxc[j+1,0]-BCxc[j,0])
                    j=len(norm)+1     # Done - leave loop
                j+=1
            x+=1

        fig = plt.figure()
        plt.title('Top Edge U Displacements')
        x=0
        for i in edgeSets['*Nset, nset=TOP,']:
            plt.scatter(i[1],TU[x])
            x+=1
        plt.plot(BCxc[:,0],BCxc[:,1])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('U Displacement (um)')
        plt.show()


        fig = plt.figure()
        plt.title('Top Edge V Displacements')
        x=0
        for i in edgeSets['*Nset, nset=TOP,']:
            plt.scatter(i[1],TV[x])
            x+=1
        plt.plot(BCxc[:,0],BCxc[:,2])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('V Displacement (um)')
        plt.show()


        #   Bottom Edge - for each node locate corresponding two points either side with DIC (in y axis)
        BV=np.zeros(len(edgeSets['*Nset, nset=BOT,']))
        BU=np.zeros(len(edgeSets['*Nset, nset=BOT,']))
        NB=[int(0) for i in range(len(edgeSets['*Nset, nset=BOT,']))]



        x=0
        for i in edgeSets['*Nset, nset=BOT,']:
            norm=BCxc[:,0]-i[1]
            j=0
            while j<len(norm):
                if all(norm<0): # If the model extends beyond the region we have displacement data for
                    NB[x]=int(i[0])
                    BU[x]=BCxc[-1,3]
                    BV[x]=BCxc[-1,4]
                    j=len(norm)     # Done - leave loop
                    print("all found negative")
                    print(norm)
                elif all(norm>0): # If model starts before the region we have displacement data for
                    NB[x]=int(i[0])
                    BU[x]=BCxc[0,3]
                    BV[x]=BCxc[0,4]
                    j=len(norm)     # Done - leave loop
                    print("all found positive")
                    print(norm)
                elif norm[j]<0 and norm[j+1]>0:
                    #print(j)
                    #print(norm[j])
                    #print(norm[j+1])
                    #print(BCyc[j,1])
                    NB[x]=int(i[0])
                    BU[x]=BCxc[j,3]+norm[j]*-1*(BCxc[j+1,3]-BCxc[j,3])/(BCxc[j+1,0]-BCxc[j,0])
                    BV[x]=BCxc[j,4]+norm[j]*-1*(BCxc[j+1,4]-BCxc[j,4])/(BCxc[j+1,0]-BCxc[j,0])
                    j=len(norm)     # Done - leave loop
                j+=1
            x+=1


        fig = plt.figure()
        plt.title('Bottom Edge U Displacements')
        x=0
        for i in edgeSets['*Nset, nset=BOT,']:
            plt.scatter(i[1],BU[x])
            x+=1
        plt.plot(BCxc[:,0],BCxc[:,3])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('U Displacement (um)')
        plt.show()


        fig = plt.figure()
        plt.title('Bottom Edge V Displacements')
        x=0
        for i in edgeSets['*Nset, nset=BOT,']:
            plt.scatter(i[1],BV[x])
            x+=1
        plt.plot(BCxc[:,0],BCxc[:,4])
        plt.xlabel('Coordinate Along Surface (um)')
        plt.ylabel('V Displacement (um)')
        plt.show()

        # Write an include file for DISP subroutine (smaller models ~<300k nodes)

        nodeList=np.concatenate((NB,NL,NR,NT))
        UdispList=np.concatenate((BU,LU,RU,TU))
        VdispList=np.concatenate((BV,LV,RV,TV))
        stringList=['      REAL*8,dimension(2,'+str(max(nodeList))+') :: nodedisp']

        x=0
        for i in nodeList:
            stringList=np.append(stringList, "      nodedisp(:,"+str(i)+") = (/"+str(round(UdispList[nodeList==i][0],3))+", "+str(round(VdispList[nodeList==i][0],3))+"/)")
            x+=1


        # for i in range(max(nodeList)):
        #     if i+1 in nodeList:
        #         stringList=np.append(stringList, "      nodedisp(:,"+str(i+1)+") = (/"+str(round(UdispList[nodeList==i+1][0],4))+", "+str(round(VdispList[nodeList==i+1][0],4))+"/)")
        #     else:
        #         stringList=np.append(stringList, "      nodedisp(:,"+str(i+1)+") = (/0.0000, 0.0000/)")


        with open("dispdata.f", "w") as output:
            output.writelines("%s\n" % i for i in stringList)

        #Write include files for Abaqus input file

        #Collate and reorder all data wrt. node number

        ndata=np.c_[nodeList,UdispList,VdispList]

        #ndata=sorted(ndata,key=lambda x: x[0])

        nodeListraw=[int(i[0]) for i in ndata]
        nodeListraw=np.array(nodeList)
        UdispListraw=[i[1] for i in ndata]
        VdispListraw=[i[2] for i in ndata]


        #Remove duplicate data
        nodeList = []
        UdispList=[]
        VdispList=[]
        for i in range(len(nodeListraw)):
            if nodeListraw[i] not in nodeList:
                nodeList.append(nodeListraw[i])
                UdispList.append(UdispListraw[i])
                VdispList.append(VdispListraw[i])

        stringListSet=[]
        for i in range(len(nodeList)):
            stringListSet=np.append(stringListSet, '*Nset, nset=NodeSet'+str(i)+', instance=Grains-1')
            stringListSet=np.append(stringListSet, str(nodeList[i])+',')

        stringListBC=[]
        for i in range(len(nodeList)):
            stringListBC=np.append(stringListBC, '*Boundary')
            stringListBC=np.append(stringListBC, 'NodeSet'+str(i)+', 1, 1, '+str(UdispList[i]))
            stringListBC=np.append(stringListBC, 'NodeSet'+str(i)+', 2, 2, '+str(VdispList[i]))

        with open("NodeSets.inc", "w") as output:
            output.writelines("%s\n" % i for i in stringListSet)

        with open("Boundaries.inc", "w") as output:
            output.writelines("%s\n" % i for i in stringListBC)



#             # Compile and save BC data:

#             data_array=np.c_[self.yc,LU,LV,RU,RV]
#             np.save('BCyc.npy',data_array)
#             df=pd.DataFrame(data_array,columns = ['yc','LU','LV','RU','RV'])
#             data_array2=np.c_[self.xc,TU,TV,BU,BV]
#             np.save('BCxc.npy',data_array2)

#             # Save boundaries to plot elsewhere:

#             np.save('Boundaries.npy',boundaries)



#             # Save data to map grain boundaries for abaqus sketch



#             np.savez('RegionData.npz', boundaries=boundaries, xc=xc, yc=yc, grains=grains, phi1=phi1,phi2=phi2, theta=theta)

# f=open('CentrlCoord.dat',"r")
# lines=f.readlines()
# f.close()

# aba_coords=np.zeros((len(lines),3))
# for x in range(len(lines)):
#     extract=lines[x][lines[x].find('(')+1:lines[x].find(')')]
#     aba_coords[x]=extract.split(',')

# #order the list of indicies to the grain number order defined in abaqus
# ordered_coords=np.zeros((len(x_ind),2))
# R_aba=np.zeros((len(aba_coords),9))
# for i in range(len(aba_coords)):
#     x_diff=grain_x-aba_coords[i,0]
#     y_diff=grain_y-aba_coords[i,1]
#     diff=abs(x_diff)+abs(y_diff)
#     ordered_coords[i,0]=grain_x[diff==min(diff)] #x coordinate
#     ordered_coords[i,1]=grain_y[diff==min(diff)] #y coordinate
#     R_aba[i,:]=R[diff==min(diff)]

# fig, current_ax = plt.subplots()
# plt.scatter(aba_coords[:,0],aba_coords[:,1], color='blue', label='Abaqus Grains')
# plt.scatter(grain_x,grain_y,color='red', marker='X', label='Experimental Grains')
# for i in range(len(aba_coords)):
#     plt.annotate(str(i+1),ordered_coords[i,:])
# plt.legend()



# # Create include file

# f = open('INP_MAT.txt', "r")

# material_data=f.readlines()

# f.close()

# include_lines=[]#np.zeros(len(material_data)*len(aba_coords))
# for i in range(len(aba_coords)):
#     material_data[1]='*MATERIAL, NAME = M'+str(i+1)
#     R_data=[str(x) for x in R_aba[i,:]]
#     material_data[11]=",".join(np.append(str(0),R_data[0:7]))+","
#     material_data[12]=",".join(R_data[7:9])+","
#     include_lines=np.append(include_lines,material_data)

# # Save the include file
# with open('material.inc', 'w') as f:
#     for item in include_lines:
#         f.write("%s\n" % item.rstrip('\n'))







# # Plotting

# grad_step = min(abs((np.diff(xc)))) # pixel spacing
# F12,F11 = np.gradient(x_map_corrected,grad_step, grad_step)
# F11=F11+1

# fig, ax = plt.subplots()
# plt.title('Deformation Gradient Component')
# ax.set_aspect(aspect=1)
# cax=plt.contourf(xc,yc,F12)
# plt.xlabel('X coordinate (um)')
# plt.ylabel('Y coordinate (um)')
# cbar = fig.colorbar(cax)

# grad_step = min(abs((np.diff(yc)))) # pixel spacing
# F22,F21 = np.gradient(y_map_corrected, grad_step, grad_step)
# F22=F22+1

# E11=0.5*(F11*F11-1)
# E22=0.5*(F22*F22-1)
# E12=0.5*(F12*F21)

# det=F11*F22-F21*F12

# maxshear=np.sqrt(((E11-E22)/2)**2+E12**2)

# fig, current_ax = plt.subplots()
# plt.title('Max Shear Plot')
# #ax.set_aspect(aspect=1)
# current_ax.axis('equal')
# plt.contour(xc,yc,boundaries,color='white')
# CS=plt.contourf(xc,yc,maxshear,512, cmap='rainbow')
# plt.xlabel('X coordinate (um)')
# plt.ylabel('Y coordinate (um)')
# plt.clim(0,0.15)
# plt.colorbar(CS)
