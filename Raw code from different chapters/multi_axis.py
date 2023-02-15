import matplotlib.pyplot as plt                 # For normal plotting
from mpl_toolkits.mplot3d import proj3d         # For 3D plotting
import numpy as np                             # For maths
from scipy import ndimage                       # For image rotations
import RegTomoReconMulti as rtr                 # Modified version of Rob's CS code
from scipy import optimize                      # For function minimization
from PIL import Image
import copy                                     # For deepcopy
try:
    import astra
    from pyevtk.hl import gridToVTK
                                        # For tomography framework
    #import transforms3d                             # For some rotation work
    
except:
    print('Astra/transforms/pyevtk import failed')

from scipy import constants  
import matplotlib.patches as patches
import matplotlib
from matplotlib.colors import ListedColormap
import ipywidgets
from libertem.utils.generate import hologram_frame
from scipy.ndimage import zoom
from skimage.restoration import unwrap_phase

def generate_tri_pris(n = 100, size_n = 1,pi=1):
    """ 
    Generate triangular prism data (with missing slice)
    
    Input:
    n = number of nodes in each dimension (nxnxn grid)
    size_n = length in nm of each node
    
    Output:
    X,Y,Z,MX,MY,MZ = Gridded coordinates, gridded magnetisation
    """
    
    # Define gradient/intercept of bounding lines
    m1, c1 = 5, 100
    m2, c2 = 0, -25
    m3, c3 = -0.6, 0
    
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Assign density
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if y < (m1*x+c1) and y > (m2*x + c2) and y < (m3*x + c3) and ((z >-20 and z<-10) or (z>0 and z<40)):
                    p = pi
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_sphere(n = 100, size_n = 1,pi=1,c=(0,0,0),r=30):
    """ Generate sphere of radius r centred at c
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

#     c = (0,0,0)
#     r = 30
    
    # Assign density to sphere
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2 < r**2:
                    p = 1
                    data.append([x,y,z,p])

                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_tetrapod(n = 100, size_n = 1,pi=1, r_tet=40,r_cyl = 10):
    """ Generate a tetrapod centred at (0,0,0), A-D labelled vertices,
    starting at top and going c/w. AOB is in the xz plane.
    Length of each leg is r_tet and radius of each leg is r_cyl """

    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Tetrahedron with O at centre, A-D labelled vertices starting at top and going c/w. AOB is in the xz plane.
    r_tet = r_tet # length of each leg of the tetrapod (i.e. length of OA, OB, OC, OD)
    c = (0,0,0) # origin of tetrapod - note changing this currently doesn't work...
    h = r_tet * (2/3)**.5 / (3/8)**.5 # height in z of the tetrapod
    
    # Calculate tetrahedron vertices
#     A = (c[0],c[1],c[2]+r_tet)
#     B = (c[0]+(r_tet**2-(h-r_tet)**2)**.5,c[1],c[2]-(h-r_tet))
#     mrot = multi_axis.rotation_matrix(0,0,120)
#     C = np.dot(mrot,B)
#     mrot = multi_axis.rotation_matrix(0,0,-120)
#     D = np.dot(mrot,B)

    # Cylinder from centre to top vertex of the tetrahedron
    r_cyl = r_cyl

    # Assign density to first cylinder
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (x-c[0])**2 + (y-c[1])**2 < r_cyl**2 and ((z >c[2] and z<(c[2]+r_tet))):
                    p = pi
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)
    P = P.reshape(len(xs),len(ys),len(zs))

    # Rotate cylinder to get other legs
    OA = rotate_bulk(P,0,0,0)
    OB = rotate_bulk(OA,0,120,0)
    OC = rotate_bulk(OB,0,0,120)
    OD = rotate_bulk(OB,0,0,-120)

    # Add all together and clip between 0 and assigned density
    tetrapod = np.clip((OA + OB + OC + OD),0,pi)
    
    return X,Y,Z,tetrapod

def generate_pillar_cavities(n = 100, size_n = 1,pi=1,x_len=70,y_len=50,z_len=50,r_cyl=15,depth=25,nx=1,ny=1,cavity_val=0):
    """ Generate box of dimensions (x_len,y_len,z_len) with hollow pillars of 'depth' length
        etched into the top z face. There will be an array of nx x ny pillars, each of radius r_cyl.
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    # Box dimensions
#     x_len = 70
#     y_len = 50
#     z_len = 50

#     # tubes
#     r_cyl = 4
#     depth = 25
#     nx = 5
#     ny = 3
    cs = []

    cxs = np.linspace(-x_len/2,x_len/2,num=nx+2)[1:-1]
    cys = np.linspace(-y_len/2,y_len/2,num=ny+2)[1:-1]

    for cx in cxs:
        for cy in cys:
            cs.append((cx,cy))

    # Assign density to box
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if -x_len/2 < x < x_len/2 and -y_len/2 < y < y_len/2 and -z_len/2 < z < z_len/2:
                    p = 1

                    for c in cs:
                        if (x-c[0])**2 + (y-c[1])**2 < r_cyl**2 and z > z_len/2-depth:
                            p = cavity_val

                    data.append([x,y,z,p])

                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def generate_layered_rod(n = 100, size_n = 1,pi=1,r=25,length=80,disc_width = 10):
    """ Generate cylindrical rod of length 80, with alternating discs every 'disc_width'
    Rod is aligned along z
    """
    # Generate x,y,z value
    xs = np.linspace(-n/2,n/2,int(n/size_n))
    ys = np.linspace(-n/2,n/2,int(n/size_n))
    zs = np.linspace(-n/2,n/2,int(n/size_n))
    X,Y,Z = np.meshgrid(xs,ys,zs,indexing='ij')

    c = (0,0,0)

    # Assign density to rod
    data = []
    for x in xs:
        for y in ys:
            for z in zs:
                if (y-c[1])**2 + (x-c[0])**2 < r**2 and -length/2 < z < length/2 :
                    p = 1
                    if np.floor(abs(-length/2-z)/disc_width)%2 == 0:
                        p = .25
                        
                    data.append([x,y,z,p])
                else:
                    p = 0
                    data.append([x,y,z,p])

    # Extract density
    P = np.take(data,3,axis=1)

    P = P.reshape(len(xs),len(ys),len(zs))
    
    return X,Y,Z,P

def plot_2d(X,Y,Z,P,s=5,size=0.1, width = 0.005, title='',ax=None,fig=None):
    """
    Plot magnetisation data in 2D
    
    Input:
    x,y = 'Projected' 2D coordinates (nxn)
    u,v = 'Projected' 2D magnetisation (nxn)
    s = Quiver plot skip density
    size = Arrow length scaling
    width = Arrow thickness 
    
    Output:
    2D Plot of magnetisation:
    - Arrows show direction of M
    - Background color shows magnitude of M
    """
    # Project along z by averaging
    x_proj = np.mean(X,axis=2)
    y_proj = np.mean(Y,axis=2)
    z_proj = np.mean(Z,axis=2)
    p_proj = np.mean(P,axis=2)
    
    if ax == None:
        # Create figure
        fig,ax = plt.subplots(figsize=(6, 8))

    # Plot magnitude
    im1 = ax.imshow(np.flipud(p_proj.T),vmin=0,vmax=1,cmap='Blues',
                     extent=(np.min(x_proj),np.max(x_proj),np.min(y_proj),np.max(y_proj)))
    
    # Add colorbar and labels
    clb = fig.colorbar(im1,ax=ax,fraction=0.046, pad=0.04)
    ax.set_xlabel('x / nm',fontsize=14)
    ax.set_ylabel('y / nm',fontsize=14)
    ax.set_title(title, fontsize= 16)

#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
    
    plt.tight_layout()
    
def rotate_bulk(P,ax,ay,az,mode='ndimage'):
    """ 
    Rotate magnetisation locations from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    
    Can use PIL or ndimage. ndimage def works but PIL faster (should work! currently doesn't handle -ve)
    
    NOTE: This implementation of scipy rotations is EXTRINSIC
    Therefore, to make it compatible with our intrinsic vector
    rotation, we swap the order of rotations (i.e. x then y then z)
    """
    # Due to indexing, ay needs reversing for desired behaviour
    if mode == 'PIL':
        nx,ny,nz = np.shape(P)
        Prot = np.zeros_like(P)
        ax,ay,az=ax,-ay,az
        scale = 256/np.max(P)
        for i in range(nx):
            im = Image.fromarray(P[i,:,:]*scale).convert('L')
            im = im.rotate(ax,resample = Image.BILINEAR)
            Prot[i,:,:] = np.array(im)/scale
        for j in range(ny):
            im = Image.fromarray(Prot[:,j,:]*scale).convert('L')
            im = im.rotate(ay,resample = Image.BILINEAR)
            Prot[:,j,:] = np.array(im)/scale
        for k in range(nz):
            im = Image.fromarray(Prot[:,:,k]*scale).convert('L')
            im = im.rotate(az,resample = Image.BILINEAR)
            Prot[:,:,k] = np.array(im)/scale
            
        return Prot
    
    else:
        ay = -ay

        P = ndimage.rotate(P,ax,reshape=False,axes=(1,2),order=1)
        P = ndimage.rotate(P,ay,reshape=False,axes=(2,0),order=1)
        P = ndimage.rotate(P,az,reshape=False,axes=(0,1),order=1)

        return P

def plot_plane(P,ax,v=[0,0,1]):
    x,y,z = v
    y = -y
    s=5
    # create x,y
    xx, yy = np.meshgrid(np.linspace(15/s,85/s,5), np.linspace(15/s,85/s,5))

    normal = [x,y,z]
    d = -np.array([50/s,50/s,50/s]).dot(normal)

    # calculate corresponding z
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]

    ax.plot_surface(xx, yy, zz, alpha=0.2,color='salmon')


    ax.plot([50/s,(50+50*x)/s],[50/s,(50+50*y)/s],[50/s,(50+50*z)/s],color='k')
    ax.plot([(50+50*x)/s],[(50+50*y)/s],[(50+50*z)/s],'o',color='red')


    im = ax.voxels(P[::s,::s,::s], facecolors=[0,0,1,.1], edgecolor=[1,1,1,0.1])



    # Add axis labels
    plt.xlabel('x / nm', fontsize=15)
    plt.ylabel('y / nm', fontsize=15)
    ax.set_zlabel('z / nm', fontsize=15)

    ax.set_xlim([0,100/s])
    ax.set_ylim([0,100/s])
    ax.set_zlim([0,100/s])

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    
def plot_both(X,Y,Z,P,ax,ay,az,save_path=None):
    # plot in 3D for a single tilt
    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    vx,vy,vz = angle_to_vector(ax,ay,az)
    Prot=rotate_bulk(P,ax,ay,az)
    plot_2d(X,Y,Z,Prot,ax=ax1,fig=fig)
    plot_plane(Prot,ax2,v=[vx,vy,vz])

    title = 'Projected density $(%i^{\circ},%i^{\circ},%i^{\circ})$' % (ax,ay,az)
    ax1.set_title(title,size=14)
    ax2.set_title('Projection direction visualised',size=14)
    
    if save_path != None:
        plt.savefig(save_path,bbox_inches='tight')
        
def angle_to_vector(ax,ay,az):
    θ = az * np.pi/180 # yaw
    ϕ = ax * np.pi/180 # pitch
    ψ = ay * np.pi/180 # roll
    
    x = -np.sin(ψ)*np.cos(θ)-np.cos(ψ)*np.sin(ϕ)*np.sin(θ)
    y = np.sin(ψ)*np.sin(θ)-np.cos(ψ)*np.sin(ϕ)*np.cos(θ)
    z = np.cos(ψ)*np.cos(ϕ)
    
    return x,y,z

def rotation_matrix(ax,ay,az,intrinsic=True):
    """ 
    Generate 3D rotation matrix from rotation angles ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """

    ax = ax * np.pi/180
    Cx = np.cos(ax)
    Sx = np.sin(ax)
    mrotx = np.array([[1,0,0],[0,Cx,-Sx],[0,Sx,Cx]])
    
    ay = ay * np.pi/180
    Cy = np.cos(ay)
    Sy = np.sin(ay)
    mroty = np.array([[Cy,0,Sy],[0,1,0],[-Sy,0,Cy]])
    
    az = az * np.pi/180
    Cz = np.cos(az)
    Sz = np.sin(az)
    mrotz = np.array([[Cz,-Sz,0],[Sz,Cz,0],[0,0,1]])
    
    if intrinsic == True:
        mrot = mrotz.dot(mroty).dot(mrotx)
    else:
        # To define mrot in an extrinsic space, matching
        # our desire for intrinsic rotation, we need
        # to swap the order of the applied rotations
        mrot = mrotx.dot(mroty).dot(mrotz)
    
    return mrot

def get_astravec(ax,ay,az):
    """ Given angles in degrees, return r,d,u,v as a concatenation
    of four 3-component vectors"""
    # Since we us flipud on y axis, ay needs reversing for desired behaviour
    ay = -ay 
    
    # centre of detector
    d = [0,0,0]
    
    # 3D rotation matrix - EXTRINSIC!
    mrot = np.array(rotation_matrix(ax,ay,az,intrinsic=False))
    
    # ray direction r
    r = mrot.dot([0,0,1])*-1 # -1 to match astra definitions
    # u (det +x)
    u = mrot.dot([1,0,0])
    # v (det +y)
    v = mrot.dot([0,1,0])

    return np.concatenate((r,d,u,v))

def generate_angles(mode='x',n_tilt = 40, alpha=70,beta=40,gamma=180,dist_n2=8,tilt2='gamma'):
    """ Return a list of [ax,ay,az] lists, each corresponding to axial
    rotations applied to [0,0,1] to get a new projection direction.
    
    Modes = x, y, dual, quad, sync, dist, rand
    
    Specify the +- tilt range of alpha/beta/gamma
    
    Say total number of tilts n_tilt
    
    For dist, each alpha has 'dist_n2' 'tilt2' projections
    
    Specify if the 2nd tilt axis is beta or gamma """
    
    angles = []
    ax,ay,az = 0,0,0
    
    # x series
    if mode=='x':
        for ax in np.linspace(-alpha,alpha,n_tilt):
            angles.append([ax,ay,az])
            
    if mode=='y':
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,n_tilt):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >= 90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,n_tilt):
                angles.append([ax,ay,az])
            
    if mode=='dual':
        for ax in np.linspace(-alpha,alpha,int(n_tilt/2)):
            angles.append([ax,ay,az])
            
        ax,ay,az = 0,0,0
        if tilt2 == 'beta':
            for ay in np.linspace(-beta,beta,int(n_tilt/2)):
                angles.append([ax,ay,az])
        if tilt2 == 'gamma':
            if gamma >=90:
                az = 90
            else:
                az = gamma
            for ax in np.linspace(-alpha,alpha,int(n_tilt/2)):
                angles.append([ax,ay,az])
    
    if mode=='quad':
        if tilt2 == 'beta':
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ax,ay,az = 0,0,0
            for ay in np.linspace(-beta,beta,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ay = beta
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
            ay = -beta
            for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                angles.append([ax,ay,az])
                    
        if tilt2 == 'gamma':
            if gamma >= 90:
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = 90
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = 45
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                    angles.append([ax,ay,az])
                az = -45
                for ax in np.linspace(-alpha,alpha,int(n_tilt/4)):
                        angles.append([ax,ay,az])           
            else:
                az = gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])
                az = -gamma/3
                for ax in np.linspace(-alpha,alpha,n_tilt/4):
                    angles.append([ax,ay,az])

    # random series # g or b
    if mode=='rand':
        for i in range(n_tilt):
            ax_rand = np.random.rand()*alpha*2 - alpha
            if tilt2 == 'beta':
                ay_rand = np.random.rand()*beta*2 - beta
                angles.append([ax_rand,ay_rand,0])
            if tilt2 == 'gamma':
                az_rand = np.random.rand()*gamma*2 - gamma
                angles.append([ax_rand,0,az_rand])
            
    # alpha propto beta series # g or b
    if mode=='sync':
        if tilt2 == 'beta': 
            ax = np.linspace(-alpha,alpha,int(n_tilt/2))
            ay = np.linspace(-beta,beta,int(n_tilt/2))

            for i,a in enumerate(ax):
                angles.append([a,ay[i],0])

            for i,a in enumerate(ax):
                angles.append([a,-ay[i],0])
        if tilt2 == 'gamma': 
            ax = np.linspace(-alpha,alpha,int(n_tilt/2))
            az = np.linspace(-gamma,gamma,int(n_tilt/2))

            for i,a in enumerate(ax):
                angles.append([a,0,az[i]])

            for i,a in enumerate(ax):
                angles.append([a,0,-az[i]])
                
        # alpha propto beta series # g or b
    if mode=='sx':
        if tilt2 == 'beta': 
            ax = np.linspace(-alpha,alpha,int(n_tilt))
            ay = np.linspace(-beta,beta,int(n_tilt))

            for i,a in enumerate(ax):
                angles.append([a,ay[i],0])

        if tilt2 == 'gamma': 
            ax = np.linspace(-alpha,alpha,int(n_tilt))
            az = np.linspace(-gamma,gamma,int(n_tilt))

            for i,a in enumerate(ax):
                angles.append([a,0,az[i]])

            # for i,a in enumerate(ax):
            #     angles.append([a,0,-az[i]])
            
    # even spacing # g or b
    if mode=='dist':
        ax = np.linspace(-alpha,alpha,int(n_tilt/dist_n2))
        if alpha == 90:
            ax = np.linspace(-90,90,n_tilt/dist_n2+1)
            ax = ax[::-1]
        if tilt2 == 'beta': 
            ay = np.linspace(-beta,beta,dist_n2)
            for x in ax:
                for y in ay:
                    angles.append([x,y,0])
        if tilt2 == 'gamma': 
            if gamma < 90:
                az = np.linspace(-gamma,gamma,dist_n2)
                for x in ax:
                    for z in az:
                        angles.append([x,0,z])
            if gamma >= 90:
                az = np.linspace(-90,90,dist_n2+1)
                for x in ax:
                    for z in az[:-1]:
                        angles.append([x,0,z])
    
    return angles

def generate_proj_data(P,angles,normalise=True):
    """ Returns projection dataset given phantom P
    and 3D projection angles list.
    
    Output is normalised and reshaped such that the
    projection slice dimension is in the middle, so as
    to be compatible with astra."""
    P_projs = []
    
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        P_rot = rotate_bulk(P,ax,ay,az) 
        P_rot_proj =np.flipud(np.sum(P_rot,axis=2).T) #flip/T match data shape to expectations
        P_projs.append(P_rot_proj) 
        
    # Prepare projections for reconstruction
    raw_data = np.array(P_projs)
    if normalise == True:
        raw_data = raw_data -  raw_data.min()
        raw_data = raw_data/raw_data.max()
    raw_data = np.transpose(raw_data,axes=[1,0,2]) # reshape so proj is middle column
        
    return raw_data
      
def generate_vectors(angles):
    """ Converts list of 3D projection angles into
    list of astra-compatible projection vectors,
    with [r,d,u,v] vectors on each row. """
    vectors = []
    for [ax,ay,az] in angles:
        vector = get_astravec(ax,ay,az)
        vectors.append(vector)
    
    return vectors

def generate_reconstruction(raw_data,vectors, algorithm = 'SIRT3D_CUDA', niter=10, weight = 0.01,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Chooise from 'SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA' or 'TV1'"""
    # Astra default algorithms
    if algorithm in ['SIRT3D_CUDA','FP3D_CUDA','BP3D_CUDA','CGLS3D_CUDA']:
        # Load data objects into astra C layer
        proj_geom = astra.create_proj_geom('parallel3d_vec',np.shape(raw_data)[0],np.shape(raw_data)[2],np.array(vectors))
        projections_id = astra.data3d.create('-sino', proj_geom, raw_data)
        vol_geom = astra.creators.create_vol_geom(np.shape(raw_data)[0], np.shape(raw_data)[0],
                                                  np.shape(raw_data)[2])
        reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
        alg_cfg = astra.astra_dict(algorithm)
        alg_cfg['ProjectionDataId'] = projections_id
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)

        astra.algorithm.run(algorithm_id,iterations=niter)
        recon = astra.data3d.get(reconstruction_id)
    
    # CS TV using RTR
    if algorithm == 'TV1':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=1)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
            
    if algorithm == 'TV1_unnorm':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV_unnorm(vol_shape, order=1)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
            
    if algorithm == 'TV_unnorm_scaled':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV_unnorm_scaled(vol_shape)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    if algorithm == 'TV2':
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.TV(vol_shape, order=2)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps=steps,callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
            
    if 'wavelet' in algorithm:
        _,w = algorithm.split(sep='_')
        data = rtr.tomo_data(raw_data, np.array(vectors), degrees=True,
                    tilt_axis=0, stack_dim=1)

        vol_shape = (data.shape[0],data.shape[0],data.shape[2])
        projector = data.getOperator(vol_shape=vol_shape,
                                    backend='astra',GPU=True)
        alg = rtr.Wavelet(vol_shape, wavelet=w)
        
        if callback_freq == 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps='classic',
                    callback=None)
            
        if callback_freq != 0:
            recon = alg.run(data=data,op=projector, maxiter=niter, weight=weight,
                    balance=balance, steps='classic',callback_freq = callback_freq,
                    callback=('primal','gap','violation','step'))[0]
    
    return recon
    
    return recon

def reorient_reconstruction(r,normalise=True):
    # Swap columns back to match orientation of phantom
    r = np.transpose(r,[2,1,0]) # Reverse column order
    r = r[:,::-1,:] # Reverse the y data
    if normalise == True:
        r = r -  r.min() # normalise
        r = r/r.max()

    recon_vector = copy.deepcopy(r)
    
    return recon_vector

def COD(P,recon):
    """ Calculate the coefficinet of determination (1 perfect, 0 shit)"""
    P_mean = np.mean(P)
    R_mean = np.mean(recon)
    sumprod = np.sum((P-P_mean)*(recon-R_mean))
    geom_mean = np.sqrt(np.sum((P-P_mean)**2)*np.sum((recon-R_mean)**2))
    coeff_norm = sumprod/geom_mean
    COD = coeff_norm**2
    
    return COD

def error_opt(beta,recon,P):
    a = np.linalg.norm(recon*beta-P)
    b = np.linalg.norm(P)
    return a/b

def phantom_error(P,recon,beta=1):
    """ Calculate normalised error between phantom and reconstruction
    (0 great, 1 shit) """
    opt = optimize.minimize(error_opt,1,args=(recon,P))
    err_phant = opt.fun
    return err_phant

def projection_error(P,recon,angles,beta=1):
    """ Calculate normalised error between phantom projections and reconstruction
    projections (0 great, 1 shit) """
    true_proj = generate_proj_data(P,angles)
    recon_proj = generate_proj_data(recon,angles)
    err_proj = phantom_error(true_proj,recon_proj,1)
    return err_proj

def noisy(image, noise_typ='gauss',g_var = 0.1, p_sp = 0.004,val_pois = None,sp_var=1):
    """ Add noise to image with choice from:
    - 'gauss' for Gaussian noise w/ variance 'g_var'
    - 's&p' for salt & pepper noise with probability 'p_sp'
    - 'poisson' for shot noise with avg count of 'val_pois'
    - 'speckle' for speckle noise w/ variance 'sp_var'"""
    if noise_typ == "gauss":
        # INDEPENDENT (ADDITIVE)
        # Draw random samples from a Gaussian distribution
        # Add these to the image
        # Higher variance = more noise
        row,col,ch= image.shape
        mean = 0
        var = g_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    
    elif noise_typ == "s&p":
        # INDEPENDENT
        # Salt & pepper/spike/dropout noise will either
        # set random pixels to their max (salt) or min (pepper)
        # Quantified by the % of corrupted pixels
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = p_sp
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.max(image) # set value to max

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape] # randomly select coordinates
        out[coords] = np.min(image) # set value to min
        return out
    
    elif noise_typ == "poisson":
        # DEPENDENT (MULTIPLICATIVE)
        # Poisson noise or shot noise arises due to the quantised
        # nature of particle detection.
        # Each pixel changes from its original value to 
        # a value taken from a Poisson distrubution with
        # the same mean (multiplied by vals)
        # So val can be thought of as the avg no. of electrons
        # contributing to that pixel of the image (low = noisy)
        if val_pois == None:
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
        else:
            vals = val_pois
            
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    
    elif noise_typ =="speckle":
        # DEPENDENT (MULTIPLICATIVE)
        # Random value multiplications of the image pixels
        
        # Generate array in shape of image but with values
        # drawn from a Gaussian distribution
        row,col,ch= image.shape
        mean = 0
        var = sp_var
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        
        # Multiply image by dist. and add to image
        noisy = image + image * gauss
        return noisy
    
def vec_to_ang(v):
    """ Returns a set of Euler rotations that map [0,0,1] to V
    Note: This will not be unique, but will be 'an' answer.
    
    https://stackoverflow.com/questions/51565760/euler-angles-and-rotation-matrix-from-two-3d-points 
    It works by first finding the axis of rotation from AxB,
    then getting the angle with atan(AxB/A.B).
    It then converts this to a rotation matrix and finally to 
    Euler angles using another module"""
    A = np.array([0,0,1])
    B = np.array(v)

    cross = np.cross(A, B)
    dot = np.dot(A, B.transpose())
    angle = np.arctan2(np.linalg.norm(cross), dot)
    rotation_axes = normalize(cross)
    rotation_m = transforms3d.axangles.axangle2mat(rotation_axes, angle, True)
    rotation_angles = transforms3d.euler.mat2euler(rotation_m, 'sxyz')
    
    return np.array(rotation_angles)*180/np.pi

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def compare_projection(recon_vector,P,ax=0,ay=0,az=0):
    """ Plots reconstruction and phantom side by side and prints error metrics """
    a = rotate_bulk(recon_vector,ax,ay,az)

    fig= plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(np.flipud(np.mean(a,axis=2).T))
    ax2.imshow(np.flipud(np.mean(rotate_bulk(P,ax,ay,az),axis=2).T))
    ax1.axis('off')
    ax2.axis('off')
    plt.tight_layout()
    ax1.set_title('Reconstruction',fontsize=14)
    ax2.set_title('Phantom',fontsize=14)

    print('Phantom error: ',phantom_error(P,recon_vector),'COD: ',COD(P,recon_vector))
    
def compare_ortho(P,r,ax=0,ay=0,az=0,ix=None,iy=None,iz=None):
    """ Plot recon orthoslices above phantom orthoslices and print error metrics"""
    
    Prot = rotate_bulk(P,ax,ay,az)
    rrot = rotate_bulk(r,ax,ay,az)
    
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(2,3,1)
    ax2 = fig.add_subplot(2,3,2)
    ax3 = fig.add_subplot(2,3,3)
    ax4 = fig.add_subplot(2,3,4)
    ax5 = fig.add_subplot(2,3,5)
    ax6 = fig.add_subplot(2,3,6)

    plot_orthoslices(rrot,axs=[ax1,ax2,ax3],ix=ix,iy=iy,iz=iz)
    plot_orthoslices(Prot,axs=[ax4,ax5,ax6],ix=ix,iy=iy,iz=iz)
    
    ax3.set_title('YZ - Recon',fontsize=15,weight='bold')
    ax2.set_title('XZ - Recon',fontsize=15,weight='bold')
    ax1.set_title('XY - Recon',fontsize=15,weight='bold')
    
    ax6.set_title('YZ - Phantom',fontsize=15,weight='bold')
    ax5.set_title('XZ - Phantom',fontsize=15,weight='bold')
    ax4.set_title('XY - Phantom',fontsize=15,weight='bold')
    
    plt.tight_layout()
    print('Phantom error: ',phantom_error(P,r),'COD: ',COD(P,r))
    
def plot_orthoslices(P,ix=None,iy=None,iz=None,axs=None):
    """ Plot xy,xz,yz orthoslices of a 3d volume
    Plots central slice by default, but slice can be specified """
    if axs == None:
        fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(1,3,1)
        ax2 = fig.add_subplot(1,3,2)
        ax3 = fig.add_subplot(1,3,3)
    else:
        ax1,ax2,ax3 = axs
        fig = plt.gcf()
        
    pmax, pmin = np.max(P),np.min(P)

    sx,sy,sz = np.shape(P)
    sx2 = int(sx/2)
    sy2 = int(sy/2)
    sz2 = int(sz/2)
    
    if ix != None:
        sx2 = ix
    if iy != None:
        sy2 = iy
    if iz != None:
        sz2 = iz

    ax3.imshow(P[sx2,:,:],cmap='Greys_r',vmax=pmax,vmin=pmin)
    ax2.imshow(P[:,sy2,:],cmap='Greys_r',vmax=pmax,vmin=pmin)
    ax1.imshow(P[:,:,sz2],cmap='Greys_r',vmax=pmax,vmin=pmin)
    
    

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    plt.tight_layout()

    ax3.set_title('YZ',fontsize=15,weight='bold')
    ax2.set_title('XZ',fontsize=15,weight='bold')
    ax1.set_title('XY',fontsize=15,weight='bold')
    
    fig.patch.set_facecolor('0.9')
    
def full_tomo(P,Pn,scheme='x',a=70,b=40,g=180,alg='TV1',tilt2='gamma',n_tilt=40,angles = None,dist_n2=8,niter=300,callback_freq=50,weight=0.01,normalise=True):
    """ Given a phantom, returns reconstructed volume (and projection data and angles)"""
    if angles == None:
        angles = generate_angles(mode=scheme,alpha=a,beta=b,gamma=g,tilt2=tilt2,dist_n2=dist_n2,n_tilt=n_tilt)
    raw_data = generate_proj_data(Pn,angles,normalise=normalise)
    vectors = generate_vectors(angles)
    recon = generate_reconstruction(raw_data,vectors,algorithm=alg,niter=niter,callback_freq=callback_freq,weight=weight)
    recon_vector = reorient_reconstruction(recon,normalise=normalise)
    try:
        astra.clear()
    except:
        pass
        
    return [recon_vector,raw_data,angles]


### magnetic stuff starts

def convertmag_T_Am(M_T):
    """ Enter 'equivalent' magnetisation in Tesla
    for a conversion to A/m.
    Can also input common materials as string, e.g. 
    'cobalt' """
    if M_T == 'cobalt':
        M_Am = 1435860
        
    else:
        M_Am = M_T / (4*np.pi*1e-7)
        
    return M_Am



class Magnetic_Phantom():
    """ Class for creating magnetic phantoms """

    def sphere(rad_m = 10*1e-9, Ms_Am = 797700, plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates uniformly magnetised sphere
            rad_m : Radius in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 + (k-ci)**2 < (rad_m/res)**2:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am
        
        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)

        return MX,MY,MZ, mesh_params
    
    def rectangle(lx_m = 80*1e-9,ly_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, p2 = (100*1e-9,100*1e-9,100*1e-9), n=(100,100,100)):
        """ Creates uniformly magnetised rectangle
            l_m = length of rectangle in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        # Initialise bounding box parameters
        p1 = (0,0,0)
        mesh_params = [p1,p2,n]
        resx = p2[0]/n[0] # resolution in m per px 
        resy = p2[1]/n[1] # resolution in m per px 
        resz = p2[2]/n[2] # resolution in m per px 
        cix = int(n[0]/2) # index of bbox centre
        ciy = int(n[1]/2) # index of bbox centre
        ciz = int(n[2]/2) # index of bbox centre

        # Initialise magnetisation arrays
        mx = np.linspace(0,p2[0],num=n[0]) * 0
        my = np.linspace(0,p2[1],num=n[1]) * 0
        mz = np.linspace(0,p2[2],num=n[2]) * 0
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if cix-.5*lx_m/resx < i < cix+.5*lx_m/resx and \
                       ciy-.5*ly_m/resy < j < ciy+.5*ly_m/resy and \
                       ciz-.5*lz_m/resz < k < ciz+.5*lz_m/resz:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am

        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params
    
    def disc_vortex(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        def vortex(x,y):
            """ Returns mx/my components for vortex state, 
            given input x and y """
            # angle between tangent and horizontal
            theta=-1*np.arctan2(x,y)
            # cosine/sine components
            C=np.cos(theta)
            S = np.sin(theta)
            return C, S
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        mx,my = vortex(i-ci,j-ci)
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                        
        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params
    
    def disc_uniform(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        MX[i,j,k] = np.cos(plan_rot*np.pi/180)*Ms_Am
                        MY[i,j,k] = np.sin(plan_rot*np.pi/180)*Ms_Am
                        
        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params
    
    def tri_pris(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates disk with c/w vortex magnetisation
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')
        
        # Define gradient/intercept of bounding lines
        m1, c1 = 5/(100*1e-9)*bbox_length_m,   100 /100*bbox_length_px
        m2, c2 = 0,                            -25 /100*bbox_length_px
        m3, c3 = -0.6/(100*1e-9)*bbox_length_m, 0

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    x = i-ci
                    y = j-ci
                    z = k-ci
                    if y < (m1*x+c1) and y > (m2*x + c2) and y < (m3*x + c3) and ((z >-20/100*bbox_length_px and z<-10/100*bbox_length_px) or (z>0 and z<30/100*bbox_length_px)):
                        MX[i,j,k] = Ms_Am
                        
        #MX = np.swapaxes(MX,0,1)
                        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params
    
    def rod(rad_m = 10*1e-9, lx_m = 60*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates uniformly magnetised cylindrical rod lying along x
            rad_m : Radius in metres
            lx_m = length of rod in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    x = i-ci
                    y = j-ci
                    z = k-ci
                    if (k-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lx_m/res < i < ci+.5*lx_m/res:
                        MX[i,j,k] = Ms_Am
                        
        #MX = np.swapaxes(MX,0,1)
                        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params
    
    def hopfion(bbox_length_m = 100*1e-9, bbox_length_px = 100, L=30, core_only=True,core_rad=27.5, core_thresh=0.7, core_innerrad=5, core_height=15, Ms_Am = 384000):
        """ Returns hopfion with height L, diameter 3L 
            Based on:  Paul Sutcliffe 2018 J. Phys. A: Math. Theor. 51 375401
            https://iopscience.iop.org/article/10.1088/1751-8121/aad521/meta
            
            Ms chosen since Sutcliffe says FeGe and 
           https://www.nature.com/articles/s41565-021-00954-9?proof=t%252Btarget%253D
            says 384 kA/m for FeGe is typical
        """

        # Define helper functions
        def calc_Omega(z,L):
            Omega = np.tan(np.pi*z/L)
            return Omega

        def calc_Xi(z,rho,L):
            Xi = (1 + (2*z/L)**2) * 1/(L*np.cos(np.pi*rho/(2*L)))
            return Xi

        def calc_Lambda(Omega,Xi,rho):
            Lambda = Xi**2*rho**2 + Omega**2/4
            return Lambda

        def calc_m(theta,rho,z,L):
            """ Calculates magnetisation using helper functions """
            Omega = calc_Omega(z,L)
            Xi = calc_Xi(z,rho,L)
            Lambda = calc_Lambda(Omega,Xi,rho)

            mx = (4*Xi*rho*(Omega*np.cos(theta)-(Lambda-1)*np.sin(theta)))/(1+Lambda)**2
            my =(4*Xi*rho*(Omega*np.sin(theta)-(Lambda-1)*np.cos(theta)))/(1+Lambda)**2
            mz = 1 - (8*Xi**2*rho**2)/(1+Lambda)**2

            return mx,my,mz

        # Convert between cartesian and cylindrical space
        def cart2cyl(x, y):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return(rho, phi)

        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')
        cent = int(bbox_length_px/2)
        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    x,y,z = i-cent,j-cent,k-cent
                    rho,theta = cart2cyl(x,y)
                    mx,my,mz = calc_m(theta,rho,z,L)
                    MX[i,j,k] = mx
                    MY[i,j,k] = my
                    MZ[i,j,k] = mz

        # Reorient y
        MY = -MY[::1,::-1,::1]

        if core_only == True:
            for i,a in enumerate(MX):
                for j,b in enumerate(a):
                    for k,c in enumerate(b):
                        x,y,z = i-cent,j-cent,k-cent

                        # Limit M to be only non-zero within hopfion core
                        # set to zero beyond outer radius
                        if x**2+y**2 > core_rad**2:
                            MX[i,j,k]=0
                            MY[i,j,k]=0
                            MZ[i,j,k]=0

                        # Set to zero above threshold and below inner radius
                        if abs(MZ[i,j,k])>core_thresh and x**2+y**2 < core_innerrad**2:
                            MX[i,j,k]=0
                            MY[i,j,k]=0
                            MZ[i,j,k]=0

                        # Set to zero beyond height limits
                        if abs(z)>core_height:
                            MX[i,j,k]=0
                            MY[i,j,k]=0
                            MZ[i,j,k]=0
                        if MZ[i,j,k]>core_thresh:
                            MX[i,j,k]=0
                            MY[i,j,k]=0
                            MZ[i,j,k]=0
                            
        MX = MX*Ms_Am
        MY = MY*Ms_Am
        MZ = MZ*Ms_Am

        return MX, MY, MZ
    
    def disc_horseshoe(rad_m = 30*1e-9, lz_m = 20*1e-9, Ms_Am = 797700, 
                  plan_rot=0, bbox_length_m = 100*1e-9, bbox_length_px = 100):
        """ Creates horseshoe magnet
            rad_m : Radius in metres
            lz_m = thickness of disc in metres
            Ms_Am : Magnetisation in A/m
            plan_rot : Direction of magnetisation, rotated in degrees ac/w from +x
            bbox_length_m : Length in metres of one side of the bounding box
            bbox_length_px : Length in pixels of one side of the bounding box """
        
        def vortex(x,y):
            """ Returns mx/my components for vortex state, 
            given input x and y """
            # angle between tangent and horizontal
            theta=-1*np.arctan2(x,y)
            # cosine/sine components
            C=np.cos(theta)
            S = np.sin(theta)
            return C, S
        
        # Initialise bounding box parameters
        p1 = (0,0,0)
        p2 = (bbox_length_m,bbox_length_m,bbox_length_m)
        n = (bbox_length_px,bbox_length_px,bbox_length_px)
        mesh_params = [p1,p2,n]
        res = bbox_length_m / bbox_length_px # resolution in m per px 
        ci = int(bbox_length_px/2) # index of bbox centre
        
        # Initialise magnetisation arrays
        mx = np.linspace(0,bbox_length_m,num=bbox_length_px) * 0
        my,mz = mx,mx
        MX, MY, MZ = np.meshgrid(mx, my, mz, indexing='ij')

        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (i-ci)**2 + (j-ci)**2 < (rad_m/res)**2 and ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        mx,my = vortex(i-ci,j-ci)
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                        
        # Assign magnetisation
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if (j-ci)>0 and abs(i-ci)<10:
                        mx,my = 0,0
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                        
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if rad_m/res>(j-ci)>0 and 9<(i-ci)<rad_m/res-1:
                        mx,my = 0,-1
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                    elif rad_m/res>(j-ci)>0 and -rad_m/res+1<(i-ci)<-9:
                        mx,my = 0,1
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                        
        for i,a in enumerate(MX):
            for j,b in enumerate(a):
                for k,c in enumerate(b):
                    if not ci-.5*lz_m/res < k < ci+.5*lz_m/res:
                        mx,my = 0,0
                        MX[i,j,k] = mx*Ms_Am
                        MY[i,j,k] = my*Ms_Am
                
                        
        
        MX=MX.astype(np.float32)
        MY=MY.astype(np.float32)
        MZ=MZ.astype(np.float32)
        
        return MX,MY,MZ, mesh_params

def plot_2d_mag(mx,my,mz=None,mesh_params=None,Ms=None,s=1):
    """ Takes x/y magnetisation projections and creates a plot
        uses quivers for direction and colour for magnitude 
        if mz is None, coloured by magnitude of x/y, else coloured by z"""
    if type(Ms) == type(None):
        Ms = np.max(np.max((mx**2+my**2)**.5))
    
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy = np.shape(mx)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    x = np.linspace(p1[0],p2[0],num=n[0])
    y = np.linspace(p1[1],p2[1],num=n[1])
    xs,ys = np.meshgrid(x,y)
    
    plt.quiver(xs[::s,::s],ys[::s,::s],mx[::s,::s].T,my[::s,::s].T,pivot='mid',scale=Ms*22,width=0.009,headaxislength=5,headwidth=4,minshaft=1.8)
    if type(mz) == type(None):
        mag = (mx**2+my**2)**.5
    else:
        mag = mz
    plt.imshow(mag.T,origin='lower',extent=[p1[0],p2[0],p1[1],p2[1]],vmin=-Ms,vmax=Ms,cmap='RdBu')
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    if type(mz) ==type(None):
        cbar.set_label('$|M_{\perp}$| / $A $', rotation=-270,fontsize=15)

    else:
        cbar.set_label('$M_z$', rotation=-270,fontsize=15)
    plt.xlabel('x / m',fontsize=15)
    plt.ylabel('y / m',fontsize=15)
    plt.show()
    
def project_along_z(U,mesh_params=None):
    """ Takes a 3D array and projects along the z component 
    It does this by multiplying each layer by its thickness
    and then summing down the axis. """
    if type(mesh_params) == type(None):
        p1 = (0,0,0)
        sx,sy,sz = np.shape(U)
        p2 = (sx,sy,sz)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # Get resolution    
    z_size = p2[2]
    z_res = z_size/n[2]
    
    # project
    u_proj = np.sum(U*z_res,axis=2)
    
    return u_proj

def calculate_A_3D(MX,MY,MZ, mesh_params=None,n_pad=100,tik_filter=0.01):
    """ Input(3D (nx,ny,nz) array for each component of M) and return
    three 3D arrays of magnetic vector potential 
    
    Note, returned arrays will remain padded since if they are used for
    projection to phase change this will make a difference. So the new
    mesh parameters are also returned
    
    """
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(MX)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
    
    # zero pad M to avoid FT convolution wrap-around artefacts
    mxpad = np.pad(MX,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mypad = np.pad(MY,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    mzpad = np.pad(MZ,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)

    # take 3D FT of M    
    ft_mx = np.fft.fftn(mxpad)
    ft_my = np.fft.fftn(mypad)
    ft_mz = np.fft.fftn(mzpad)
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 

    kx = np.fft.fftfreq(ft_mx.shape[0],d=resx)
    ky = np.fft.fftfreq(ft_my.shape[0],d=resy)
    kz = np.fft.fftfreq(ft_mz.shape[0],d=resz)
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij') # Create a grid of coordinates
    
    # vacuum permeability
    mu0 = 4*np.pi*1e-7
    
    # Calculate 1/k^2 with Tikhanov filter
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2+KZ**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2+KZ**2)**.5 + tik_filter*resx)**-2
    
    # M cross K
    cross_x = ft_my*KZ - ft_mz*KY
    cross_y = -ft_mx*KZ + ft_mz*KX
    cross_z = -ft_my*KX + ft_mx*KY
    
    # Calculate A(k)
    ft_Ax = (-1j * mu0 * K2_inv) * cross_x
    ft_Ay = (-1j * mu0 * K2_inv) * cross_y
    ft_Az = (-1j * mu0 * K2_inv) * cross_z
    
    # Inverse fourier transform
    Ax = np.fft.ifftn(ft_Ax)
    AX = Ax.real
    Ay = np.fft.ifftn(ft_Ay)
    AY = Ay.real
    Az = np.fft.ifftn(ft_Az)
    AZ = Az.real
    
    # new mesh parameters (with padding)
    n = (n[0]+2*n_pad,n[1]+2*n_pad,n[2]+2*n_pad)
    p2 = (p2[0]+2*n_pad*resx,p2[1]+2*n_pad*resy,p2[2]+2*n_pad*resz)
    mesh_params=(p1,p2,n)
    
    AX=AX.astype(np.float32)
    AY=AY.astype(np.float32)
    AZ=AZ.astype(np.float32)
    
    return AX,AY,AZ,mesh_params

def calculate_phase_AZ(AZ,mesh_params=None):
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy,sz = np.shape(MX)
        p2 = (sx,sy,sx)
        n = p2
        mesh_params = [p1,p2,n]
    else:
        p1,p2,n = mesh_params
    """ Calculates projected phase change from 3D AZ """
    AZ_proj = project_along_z(AZ,mesh_params=mesh_params) 
    phase = AZ_proj * -1* np.pi/constants.codata.value('mag. flux quantum') / (2*np.pi)
    return phase

def calculate_phase_M_2D(MX,MY,MZ,mesh_params,n_pad=500,tik_filter=0.01,unpad=True):
    """ Preffered method. Takes 3D MX,MY,MZ magnetisation arrays
    and calculates phase shift in rads in z direction.
    First projects M from 3D to 2D which speeds up calculations """
    p1,p2,n=mesh_params
    
    # J. Loudon et al, magnetic imaging, eq. 29
    const = .5 * 1j * 4*np.pi*1e-7 / constants.codata.value('mag. flux quantum')

    # Define resolution from mesh parameters
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    # Project magnetisation array
    mx = project_along_z(MX,mesh_params=mesh_params)
    my = project_along_z(MY,mesh_params=mesh_params)
    
    # Take fourier transform of M
    # Padding necessary to stop Fourier convolution wraparound (spectral leakage)
    if n_pad > 0:
        mx = np.pad(mx,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
        my = np.pad(my,[(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    
    ft_mx = np.fft.fft2(mx)
    ft_my = np.fft.fft2(my)
    
    # Generate K values
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    KX, KY = np.meshgrid(kx,ky, indexing='ij') # Create a grid of coordinates
    
    # Filter to avoid division by 0
    if tik_filter == 0:
        K2_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-2)
    else:
        K2_inv = ((KX**2+KY**2)**.5 + tik_filter*resx)**-2

    # Take cross product (we only need z component)
    cross_z = (-ft_my*KX + ft_mx*KY)*K2_inv
    
    # Inverse fourier transform
    phase = np.fft.ifft2(const*cross_z).real
    
    # Unpad
    if unpad == True:
        if n_pad > 0:
            phase=phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def calculate_phase_M_3D(MX,MY,MZ,mesh_params,n_pad=100,tik_filter=0.01):
    """ Slower than 2D but good for comparison. Takes 3D MX,MY,MZ magnetisation arrays
    and calculates phase shift in rads in z direction.
    Calculations performed directly in 3D """
    p1,p2,n=mesh_params
    
    # constant prefactor
    const = .5*1j*4*np.pi*1e-7/constants.codata.value('mag. flux quantum')

    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    MX = np.pad(MX,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    MY = np.pad(MY,[(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    MZ = np.pad(MZ, [(n_pad,n_pad),(n_pad,n_pad),(n_pad,n_pad)], mode='constant', constant_values=0)
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    kz = np.fft.fftfreq(n[2]+2*n_pad,d=resz)
    KX, KY, KZ = np.meshgrid(kx,ky,kz, indexing='ij') # Create a grid of coordinates
    K2_inv = np.nan_to_num(((KX**2+KY**2+KZ**2)**.5+ tik_filter*resx)**-2)
    
    # Take 3D fourier transforms (only need x and y for cross-z)
    ft_mx = np.fft.fftn(MX)
    ft_my = np.fft.fftn(MY)
    
    # take cross product
    cross_z = (-ft_my*KX + ft_mx*KY) * K2_inv
    
    # extract central slice
    slice_z = cross_z[:,:,0] * resz 
    
    # inverse fourier transform
    phase = np.fft.ifft2(const*slice_z).real
    
    if n_pad > 0:
        phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def analytical_sphere(B0_T=1.6,r_m=50*1e-9,mesh_params=None,beta=-90,n_pad=100):
    """ Analytically calculates the phase change for a sphere (from Beleggia and Zhu 2003)"""
    import scipy
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    # Calculate prefactor
    const = 4 * np.pi**2 * 1j * B0_T  *(r_m/2/np.pi)**2/ constants.codata.value('mag. flux quantum')
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    
    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)#/(2*np.pi))
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)#/(2*np.pi))
    KX, KY = np.meshgrid(kx,ky)#,indexing='ij') # Create a grid of coordinates
    
    # Calculate 1/k^2 with Tikhanov filter
    K3_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-3)
    K =(KX**2+KY**2)**.5

    #The normalized sinc function is the Fourier transform of the rectangular function with no scaling. np default is normalised
    phase_ft = const * (KY*np.cos(beta*np.pi/180) - KX*np.sin(beta*np.pi/180)) * K3_inv \
                        * scipy.special.spherical_jn(1,r_m*(2*np.pi)*K) / (resx*resy)
    
    phase = np.fft.ifft2(phase_ft).real 
    
    phase = np.fft.ifftshift(phase) 
    
    phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def analytical_rectangle(B0_T=1.6,lx_m=200*1e-9,ly_m=140*1e-9,lz_m=20*1e-9,mesh_params=None,beta=300,n_pad=100):
    """ Analytically calculates the phase change for a rectangle (from Beleggia and Zhu 2003)"""
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    # Calculate prefactor
    V = lx_m*ly_m*lz_m
    const = 1j*np.pi*B0_T*V/constants.codata.value('mag. flux quantum')
    
    # Generate K values
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 

    kx = np.fft.fftfreq(n[0]+2*n_pad,d=resx)
    ky = np.fft.fftfreq(n[1]+2*n_pad,d=resy)
    KX, KY = np.meshgrid(kx,ky,indexing='ij') # Create a grid of coordinates
    #KX,KY=KX*(2*np.pi),KY*(2*np.pi)
    
    # Calculate 1/k^2 with Tikhanov filter
    K2_inv = np.nan_to_num(((KX**2+KY**2)**.5)**-2)

    #The normalized sinc function is the Fourier transform of the rectangular function with no scaling. np default is normalised
    phase_ft = const * K2_inv * (KY*np.cos(beta*np.pi/180) - KX*np.sin(beta*np.pi/180)) * np.sinc(lx_m*KX) * np.sinc(ly_m*KY) / (resx*resy)
    
    phase = np.fft.ifft2(phase_ft).real
    
    phase = np.fft.ifftshift(phase) / (2*np.pi)
    
    if n_pad>0:
        phase = phase[n_pad:-n_pad,n_pad:-n_pad]
    
    return phase

def linsupPhi(mx=1.0, my=1.0, mz=1.0, Dshp=None, theta_x=0.0, theta_y=0.0, pre_B=1.0, pre_E=1, v=1, multiproc=True):
    """Applies linear superposition principle for 3D reconstruction of magnetic and electrostatic phase shifts.
    This function will take 3D arrays with Mx, My and Mz components of the 
    magnetization, the Dshp array consisting of the shape function for the 
    object (1 inside, 0 outside), and the tilt angles about x and y axes to 
    compute the magnetic and the electrostatic phase shift. Initial computation 
    is done in Fourier space and then real space values are returned.
    Args: 
        mx (3D array): x component of magnetization at each voxel (z,y,x)
        my (3D array): y component of magnetization at each voxel (z,y,x)
        mz (3D array): z component of magnetization at each voxel (z,y,x)
        Dshp (3D array): Binary shape function of the object. Where value is 0,
            phase is not computed.  
        theta_x (float): Rotation around x-axis (degrees). Rotates around x axis
            then y axis if both are nonzero. 
        theta_y (float): Rotation around y-axis (degrees) 
        pre_B (float): Numerical prefactor for unit conversion in calculating 
            the magnetic phase shift. Units 1/pixels^2. Generally 
            (2*pi*b0*(nm/pix)^2)/phi0 , where b0 is the Saturation induction and 
            phi0 the magnetic flux quantum. 
        pre_E (float): Numerical prefactor for unit conversion in calculating the 
            electrostatic phase shift. Equal to sigma*V0, where sigma is the 
            interaction constant of the given TEM accelerating voltage (an 
            attribute of the microscope class), and V0 the mean inner potential.
        v (int): Verbosity. v >= 1 will print status and progress when running
            without numba. v=0 will suppress all prints. 
        mp (bool): Whether or not to implement multiprocessing. 
    Returns: 
        tuple: Tuple of length 2: (ephi, mphi). Where ephi and mphi are 2D numpy
        arrays of the electrostatic and magnetic phase shifts respectively. 
    """
    import time
    vprint = print if v>=1 else lambda *a, **k: None
    [dimz,dimy,dimx] = mx.shape
    dx2 = dimx//2
    dy2 = dimy//2
    dz2 = dimz//2

    ly = (np.arange(dimy)-dy2)/dimy
    lx = (np.arange(dimx)-dx2)/dimx
    [Y,X] = np.meshgrid(ly,lx, indexing='ij')
    dk = 2.0*np.pi # Kspace vector spacing
    KX = X*dk
    KY = Y*dk
    KK = np.sqrt(KX**2 + KY**2) # same as dist(ny, nx, shift=True)*2*np.pi
    zeros = np.where(KK == 0)   # but we need KX and KY later. 
    KK[zeros] = 1.0 # remove points where KK is zero as will divide by it

    # compute S arrays (will apply constants at very end)
    inv_KK =  1/KK**2
    Sx = 1j * KX * inv_KK
    Sy = 1j * KY * inv_KK
    Sx[zeros] = 0.0
    Sy[zeros] = 0.0
    
    # Get indices for which to calculate phase shift. Skip all pixels where
    # thickness == 0 
    if Dshp is None: 
        Dshp = np.ones(mx.shape)
    # exclude indices where thickness is 0, compile into list of ((z1,y1,x1), (z2,y2...
    zz, yy, xx = np.where(Dshp != 0)
    inds = np.dstack((zz,yy,xx)).squeeze()

    # Compute the rotation angles
    st = np.sin(np.deg2rad(theta_x))
    ct = np.cos(np.deg2rad(theta_x))
    sg = np.sin(np.deg2rad(theta_y))
    cg = np.cos(np.deg2rad(theta_y))

    x = np.arange(dimx) - dx2
    y = np.arange(dimy) - dy2
    z = np.arange(dimz) - dz2
    Z,Y,X = np.meshgrid(z,y,x, indexing='ij') # grid of actual positions (centered on 0)

    # compute the rotated values; 
    # here we apply rotation about X first, then about Y
    i_n = Z*sg*ct + Y*sg*st + X*cg
    j_n = Y*ct - Z*st

    mx_n = mx*cg + my*sg*st + mz*sg*ct
    my_n = my*ct - mz*st

    # setup 
    mphi_k = np.zeros(KK.shape,dtype=complex)
    ephi_k = np.zeros(KK.shape,dtype=complex)

    nelems = np.shape(inds)[0]
    stime = time.time()
    vprint(f'Beginning phase calculation for {nelems:g} voxels.')
    if multiproc:
        vprint("Running in parallel with numba.")
        ephi_k, mphi_k = exp_sum(mphi_k, ephi_k, inds, KY, KX, j_n, i_n, my_n, mx_n, Sy, Sx)        

    else:
        vprint("Running on 1 cpu.")
        otime = time.time()
        vprint('0.00%', end=' .. ')
        cc = -1
        for ind in inds:
            ind = tuple(ind)
            cc += 1
            if time.time() - otime >= 15:
                vprint(f'{cc/nelems*100:.2f}%', end=' .. ')
                otime = time.time()
            # compute the expontential summation
            sum_term = np.exp(-1j * (KY*j_n[ind] + KX*i_n[ind]))
            ephi_k += sum_term 
            mphi_k += sum_term * (my_n[ind]*Sx - mx_n[ind]*Sy)
        vprint('100.0%')

    vprint(f"total time: {time.time()-stime:.5g} sec, {(time.time()-stime)/nelems:.5g} sec/voxel.")
    #Now we have the phases in K-space. We convert to real space and return
    ephi_k[zeros] = 0.0
    mphi_k[zeros] = 0.0
    ephi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ephi_k)))).real*pre_E
    mphi = (np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(mphi_k)))).real*pre_B

    return (ephi,mphi)

def plot_phase_proj(phase,mesh_params=None,ax=None):
    """ Plots the projected phase shift in rads """
    if mesh_params == None:
            p1 = (0,0,0)
            sx,sy = np.shape(phase)
            p2 = (sx,sy,sx)
            n = p2
    else:
        p1,p2,n = mesh_params
        
    if ax == None:
        fig,ax = plt.subplots()
    fig=plt.gcf()

    im = ax.imshow(phase.T,extent=[p1[0],p2[0],p1[1],p2[1]],origin='lower')
    cbar = fig.colorbar(im,fraction=0.046, pad=0.04,ax=ax)

    cbar.set_label('Projected phase shift / rad', rotation=-270,fontsize=15)
    ax.set_xlabel('x / m',fontsize=14)
    ax.set_ylabel('y / m',fontsize=14)
    #plt.show()
    
def calculate_B_from_A(AX,AY,AZ,mesh_params=None):
    """ Takes curl of B to get A """
    # Initialise parameters
    phase_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(AX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    p1,p2,n=mesh_params
    
    resx = p2[0]/n[0] # resolution in m per px 
    resy = p2[1]/n[1] # resolution in m per px 
    resz = p2[2]/n[2] # resolution in m per px 
    
    BX = np.gradient(AZ,resy)[1] - np.gradient(AY,resz)[2]
    BY = np.gradient(AX,resz)[2] - np.gradient(AZ,resx)[0]
    BZ = np.gradient(AY,resx)[0] - np.gradient(AX,resy)[1]
    
    BX=BX.astype(np.float32)
    BY=BY.astype(np.float32)
    BZ=BZ.astype(np.float32)
        
    return BX/(2*np.pi),BY/(2*np.pi),BZ/(2*np.pi)

def calculate_B_from_phase(phase_B,mesh_params=None):
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy = np.shape(mx)
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    x_size = p2[0]
    x_res = x_size/n[0]
    
    y_size = p2[1]
    y_res = y_size/n[1]
    
    d_phase = np.gradient(phase_B,x_res)
    b_const = (constants.codata.value('mag. flux quantum')/(np.pi))
    b_field_x = -b_const*d_phase[1]
    b_field_y = b_const*d_phase[0]

    mag_B = np.hypot(b_field_x,b_field_y)
    
    return mag_B,b_field_x,b_field_y

def plot_colorwheel(alpha=1,rot=0,flip= False, ax=None,rad=0.5,clip=48,shape=200,shift_centre=None,mesh_params=None):
    """ Plots a colorwheel
    alpha - match alpha to the alpha in your B plot (i.e. how black is the centre)
    rot - rotate the color wheel, enter angle in degrees (must be multiple of 90)
    flip - change between cw / ccw
    ax - pass an axis to plot on top of another image
    rad - radius of the colorwheel, scaled 0 to 1
    clip - radius of colorwheel clip in distance (m)
    shape - length of array size (must be square array)
    shift_centre - tuple lets you shift centre in px (horz,vert)
    """
    def cart2pol(x, y):
        """ Convert cartesian to polar coordinates
        rho = magnitude, phi = angle """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    if mesh_params == None:
        p1 = (0,0,0)
        sx,sy =200,200
        p2 = (sx,sy,sx)
        n = p2
    else:
        p1,p2,n = mesh_params
        
    extent = (p1[0],p2[0],p1[1],p2[1])
    
    ax_tog = 1
    if type(ax) == type(None):
        fig,ax = plt.subplots()
        ax_tog=0
        
    scale = (p2[0]-p1[0])
    centre = np.array((scale/2, scale/2)) #* scale/shape/100
    if type(shift_centre) == type(None):
        shift_centre=(0,0)
    shift_centre = np.array(shift_centre)/shape #* scale/shape/100
    
    # Create coordinate space
    x = np.linspace(-scale/2,scale/2,shape)
    y = x
    X,Y = np.meshgrid(x,y)

    # Map theta values onto coordinate space 
    thetas = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            # shifting will shift the centre of divergence
            thetas[ix,iy] = cart2pol((xx+shift_centre[0]*scale),(yy+shift_centre[1]*scale))[1]

    # Plot hsv colormap of angles
    if flip == False:
        im1 = ax.imshow(ndimage.rotate(thetas.T,180+rot),cmap='hsv_r',origin='lower',extent=extent,zorder=2)
    if flip == True:
        im1 = ax.imshow(ndimage.rotate(thetas,270+rot),cmap='hsv_r',origin='lower',extent=extent,zorder=2)

    # Create alpha contour map
    my_cmap = alpha_cmap()
    
    # Map circle radii onto xy coordinate space
    circ = np.ones_like(X)*0
    for ix, xx in enumerate(x):
        for iy, yy in enumerate(y):
            if (xx+shift_centre[1]*scale)**2 + (yy-shift_centre[0]*scale)**2 < (rad*scale)**2:
                #print(xx,shift_centre[0])
                circ[ix,iy] = cart2pol((xx+shift_centre[1]*scale),(yy-shift_centre[0]*scale))[0]

    # Plot circle
    im2 = plt.imshow(circ, cmap=my_cmap,alpha=alpha,extent=extent,zorder=2)
    
    print(type(ax))
    if ax_tog==1:
        # Clip to make it circular
        print(centre )#+shift_centre*scale*[-1,-1]+scale/shape/2)
        patch = patches.Circle(centre +shift_centre*scale*[1,1], radius=clip, transform=ax.transData)
        im2.set_clip_path(patch)
        im1.set_clip_path(patch)
    
    if ax_tog==0:
        # Clip to make it circular
        patch = patches.Circle(centre +shift_centre*scale*[1,1]-scale/shape/4, radius=clip, transform=ax.transData)
        im2.set_clip_path(patch)
        im1.set_clip_path(patch)
        ax.axis('off')
        
def plot_2d_B(bx,by,mesh_params=None, ax=None,s=5,scale=7,mag_res=5, quiver=True, B_contour=True,phase=None,phase_res=np.pi/50):
    """ Plot projected B field
    quiver = turn on/off the arrows
    s = quiver density
    scale = quiver arrow size
    B_contour = turn on/off |B| contour lines
    mag_res = spacing of |B| contour lines in nT
    phase = pass phase shifts to plot phase contours
    phase_res = spacing of phase contours in radians
    """
    if ax == None:
        fig,ax = plt.subplots()
    
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(bx)
        p2 = (s[0],s[1],s[0])
        n = p2
        mesh_params = [p1,p2,n]
    
    p1,p2,n = mesh_params
    mag_B = np.hypot(bx,by)

    # Create alpha contour map
    my_cmap = alpha_cmap()

    # plot B field direction as a colour
    # using tan-1(vy/vx)
    angles = np.arctan2(by,bx)
    angles = shift_angles(angles,np.pi)
    ax.imshow(angles.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]], cmap='hsv')

    # Plot magnitude of B as in black/transparent scale
    ax.imshow(mag_B.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]],interpolation='spline16', cmap=my_cmap,alpha=1)

    ax.set_xlabel('x / m', fontsize = 16)
    ax.set_ylabel('y / m', fontsize = 16)
    
    # Quiver plot of Bx,By
    if quiver==True:
        x = np.linspace(p1[0],p2[0],num=n[0])
        y = np.linspace(p1[1],p2[1],num=n[1])
        xs,ys = np.meshgrid(x,y)
        ax.quiver(xs[::s,::s],ys[::s,::s],bx[::s,::s].T,by[::s,::s].T,color='white',scale=np.max(abs(mag_B))*scale,
                  pivot='mid',width=0.009,headaxislength=5,headwidth=4,minshaft=1.8)
    
    # Contour plot of |B|
    if B_contour==True:
        mag_range = (np.max(mag_B)-np.min(mag_B))/1e-9
        n_levels = int(mag_range/mag_res)
        cs = ax.contour(mag_B.T,origin='lower',levels=10, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
        
    # Contour plot of phase
    if type(phase)!=type(None):
        phase_range = (np.max(phase)-np.min(phase))/1e-9
        n_levels = int(phase_range/phase_res)
        cs = ax.contour(phase.T-np.min(phase).T,origin='lower',levels=10, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
        
def alpha_cmap():
    """ Returns a colormap object that is black,
    with alpha=1 at vmin and alpha=0 at vmax"""
    # Create a colour map which is just black
    colors = [(0.0, 'black'), (1.0, 'black')]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("alpha_cmap", colors)
    # Get colors from current map for values 1 to 256
    # These will all be black with alpha=1 (opaque), ie. [0,0,0,1]
    my_cmap = cmap(np.arange(cmap.N))
    # Set alpha (opaque and black (1) at vmin and fully transparanet at vmax)
    my_cmap[:,-1] = np.linspace(1,0,cmap.N)
    # create new colormap with the new alpha values
    my_cmap = ListedColormap(my_cmap)
    
    return my_cmap

def shift_angles(vals,angle=None):
    """ Takes angles currently in -pi to +pi range,
    and lets you shift them by an angle, keeping them
    in the same range."""
    
    if angle == None:
        return vals
    
    newvals = vals+angle
    for i,vv in enumerate(newvals):
        for j,v in enumerate(vv):
            if v > np.pi:
                newvals[i,j] = v - 2*np.pi
            if v < -np.pi:
                newvals[i,j] = v + 2*np.pi
            
    return newvals

#### from magnetic_reconstruction
def generate_phase_data(MX,MY,MZ,angles,mesh_params=None,n_pad=500,unpad=False):
    """ Returns phase projections for given M and angles
    in order [x, i_tilt, y] """
    # Initialise parameters
    phase_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(MX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    
    # Loop through projection angles
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        #rotate M
        MXr,MYr,MZr = rotate_magnetisation(MX,MY,MZ,ax,ay,az)
        #calculate phase
        phase = calculate_phase_M_2D(MXr,MYr,MZr,mesh_params=mesh_params,n_pad=n_pad,unpad=unpad)
        phase = np.flipud(phase.T)

        phase_projs.append(phase)            
    
    # Prepare projections for reconstruction
    phase_projs = np.transpose(phase_projs,axes=[1,0,2]) # reshape so proj is middle column
    phase_projs=phase_projs.astype(np.float32)
    return np.array(phase_projs)

def rotate_magnetisation(U,V,W,ax=0,ay=0,az=0):
    """ 
    Takes 3D gridded magnetisation values as input
    and returns them after an intrinsic rotation ax,ay,az 
    about the x,y,z axes (given in degrees) 
    (Uses convention of rotating about z, then y, then x)
    """
    # Rotate the gridded locations of M values
    Ub = rotate_bulk(U,ax,ay,az)
    Vb = rotate_bulk(V,ax,ay,az)
    Wb = rotate_bulk(W,ax,ay,az)
    
    shape = np.shape(Ub)
    
    # Convert gridded values to vectors
    coor_flat = grid_to_coor(Ub,Vb,Wb)
    
    # Rotate vectors
    coor_flat_r = rotate_vector(coor_flat,ax,ay,az)
    
    # Convert vectors back to gridded values
    Ur,Vr,Wr = coor_to_grid(coor_flat_r,shape=shape)
    
    # Set small values to 0
    # (In theory the magnitude of M in each cell should be Ms,
    #  so we can set magnitude lower than this to zero -
    #  typically python rounding errors lead to very small values,
    #  which it is useful to exclude here)
#    mag_max = (np.max(U)**2+np.max(V)**2+np.max(W)**2)**0.5
#    mag = (Ur**2+Vr**2+Wr**2)**.5
#     for M in [Ur,Vr,Wr]:
#         M[abs(M)<1e-5*mag_max] = 0
#         M[mag<.6*mag_max] = 0
    
    return Ur,Vr,Wr

def grid_to_coor(U,V,W):
    """ Convert gridded 3D data (3,n,n,n) into coordinates (n^3, 3) """
    coor_flat = []
    nx = np.shape(U)[0]
    ny = np.shape(U)[1]
    nz = np.shape(U)[2]
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = U[ix,iy,iz]
                y = V[ix,iy,iz]
                z = W[ix,iy,iz]
                coor_flat.append([x,y,z])
                
    return coor_flat

def coor_to_grid(coor_flat,shape=None):
    """ Convert coordinates (n^3, 3) into gridded 3D data (3,n,n,n) """
    if shape == None:
        n = int(np.round(np.shape(coor_flat)[0]**(1/3)))
        shape = (n,n,n)
    nx,ny,nz = shape
    
    x = np.take(coor_flat,0,axis=1)
    y = np.take(coor_flat,1,axis=1)
    z = np.take(coor_flat,2,axis=1)
    U = x.reshape((nx,ny,nz))
    V = y.reshape((nx,ny,nz))
    W = z.reshape((nx,ny,nz))

    return U, V, W

def rotate_vector(coor_flat,ax,ay,az):
    """ Rotates vectors by specified angles ax,ay,az 
    about the x,y,z axes (given in degrees) """
    
    # Get rotation matrix
    mrot = rotation_matrix(ax,ay,az)    

    coor_flat_r = np.zeros_like(coor_flat)
    
    # Apply rotation matrix to each M vector
    for i,M in enumerate(coor_flat):
        coor_flat_r[i] = mrot.dot(M)
    
    return coor_flat_r

def dual_axis_phase_generation(MX,MY,MZ,mesh_params,n_tilt=40, a_range=70,n_pad = 100):
    """ Returns ax,ay, px,py (angles and phase projections from x and y tilt series)
    n_tilt = number of projections in each series
    a_range = maximum tilt angle (applies to both series)
    n_pad = padding applied during phase calculation (should be > 2x n_px) """
    angles_x = generate_angles(mode='x',n_tilt=n_tilt,alpha=a_range)
    angles_y = generate_angles(mode='y',n_tilt=n_tilt,beta=a_range,tilt2='beta')
    phases_x = generate_phase_data(MX,MY,MZ,angles_x,mesh_params=mesh_params,n_pad=n_pad,unpad=False)
    phases_y = generate_phase_data(MX,MY,MZ,angles_y,mesh_params=mesh_params,n_pad=n_pad,unpad=False)
    
    return angles_x, angles_y, phases_x, phases_y

def dual_axis_B_generation(pxs,pys,mesh_params):
    """ Returns bxs, bys (projected B fields for tilt series) 
    Calculates the BX/BY component from the y/x tilt series
    """
    # x tilt series --> derivative in x is good --> gives BY
    # above is bullshit? tilt around X --> sensitive to BX
    p1,p2,n = mesh_params
        
    x_size = p2[0]
    x_res = x_size/n[0]
    
    b_const = (constants.codata.value('mag. flux quantum')/(np.pi))
    bxs = []
    # calculate b component at each tilt
    for i in range(np.shape(pxs)[1]):
        p=pxs[:,i,:]
        # calculate_B_from_phase assumes input is ordered in the wierd way (i.e. flip.T)
        # but pxs/pys are somehow in the correct orientation, so we need to put them back for this to work
        # since gradient[0] gives the column gradient and [1] gives the row, so we'll get an incorrect answer
        #p = np.flipud(p).T
        
        # minus not needed as it goes bottom to top instead of top to bottom
        bx = b_const*np.gradient(p,x_res)[0]
        #_,bx,_ = ma.calculate_B_from_phase(p,mesh_params=mesh_params)
        bxs.append(bx)
    
    bys = []
    # calculate b component at each tilt
    for i in range(np.shape(pys)[1]):
        p=pys[:,i,:]
        #p = np.flipud(p).T
        by = b_const*np.gradient(p,x_res)[1]
        #_,_,by = ma.calculate_B_from_phase(p,mesh_params=mesh_params)
        bys.append(by)
    
    # reorder for tomo
    bxs = np.transpose(bxs,axes=[1,0,2])
    bys = np.transpose(bys,axes=[1,0,2])
    
    return bxs,bys

def dual_axis_reconstruction(xdata,ydata,axs,ays,mesh_params,algorithm = 'SIRT3D_CUDA', niter=100, weight = 0.001,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Perform reconstruction of X/Y components on either phase or magnetic projections from dual axis series """
    p1,p2,nn=mesh_params
    resx=p2[0]/nn[0]
    resy=p2[1]/nn[1]

    # X series reconstruction
    vecs = generate_vectors(axs)
    rx = generate_reconstruction(xdata,vecs, algorithm = algorithm, niter=niter, weight = weight,
                                balance = balance, steps = steps, callback_freq = callback_freq)
    
    astra.clear()
    
    # Y series reconstruction
    vecs = generate_vectors(ays)
    ry = generate_reconstruction(ydata,vecs, algorithm = algorithm, niter=niter, weight =weight,
                                balance = balance, steps = steps, callback_freq =callback_freq)
    
    # Restructure data to match input M/A/B/p
    rx = np.transpose(rx,axes=(2,1,0))[:,::-1,:]
    ry = np.transpose(ry,axes=(2,1,0))[:,::-1,:]

    rx = rx/resx
    ry = ry/resy
    
    astra.clear()
    return rx,ry

def dual_axis_bz_from_bxby(bx,by):
    #dz = -1*(np.gradient(bx)[0]+np.gradient(by)[1])
    bz = []
    old=0
    for i in range(np.shape(bx)[2]):
        dbx = bx[:,:,i]
        dby = by[:,:,i]
        dbz = -1*(np.gradient(dbx)[0]+np.gradient(dby)[1])
        cum = dbz+old
        bz.append(cum)
        old=cum    
    #bz -= np.mean(bz)
    #removed the minus
    bz = np.array(bz)
    
    bz = np.transpose(bz,axes=[1,2,0])
    return bz

def plot_3d_B_slice(bx,by,bz,i_slice=None,mesh_params=None, ax=None,s=5,scale=7,mag_res=0.1, quiver=True, cbar=False,B_contour=True,phase=None,phase_res=np.pi/50):
    """ Plot projected B field
    quiver = turn on/off the arrows
    s = quiver density
    scale = quiver arrow size
    B_contour = turn on/off |B| contour lines
    mag_res = spacing of |B| contour lines in T
    phase = pass phase shifts to plot phase contours
    phase_res = spacing of phase contours in radians
    """
    if ax == None:
        fig,ax = plt.subplots()
    
    if mesh_params == None:
        p1 = (0,0,0)
        sh = np.shape(bx)
        p2 = (sh[0],sh[1],sh[0])
        n = p2
        mesh_params = [p1,p2,n]
        
    if type(i_slice) is type(None):
        i_slice=int(np.shape(bz)[2]/2)
        
    bx=copy.deepcopy(bx)
    by=copy.deepcopy(by)
    bz=copy.deepcopy(bz)
        
    bmax = np.max((bx**2+by**2+bz**2)**0.5)
    mag_B = ((bx**2+by**2)**0.5)[:,:,i_slice]
    bx=bx[:,:,i_slice]
    by=by[:,:,i_slice]
    bz=bz[:,:,i_slice]
        
    
    p1,p2,n = mesh_params
    mag_B = np.hypot(bx,by)

    

    # plot BZ a colour
    # using tan-1(vy/vx)
    im=ax.imshow(bz.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]], cmap='RdBu',vmin=-bmax,vmax=bmax)
    if cbar==True:
        fig=plt.gcf()
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('$B_z$ / T', rotation=-270,fontsize=15)

    # Plot magnitude of B as in black/transparent scale
    # Create alpha contour map
    my_cmap = alpha_cmap()
    ax.imshow(mag_B.T,origin='lower', 
               extent=[p1[0], p2[0], p1[1],p2[1]],interpolation='spline16', cmap=my_cmap,alpha=1,vmin=0,vmax=bmax)

    ax.set_xlabel('x / m', fontsize = 16)
    ax.set_ylabel('y / m', fontsize = 16)
    
    # Quiver plot of Bx,By
    if quiver==True:
        x = np.linspace(p1[0],p2[0],num=n[0])
        y = np.linspace(p1[1],p2[1],num=n[1])
        xs,ys = np.meshgrid(x,y)
        ax.quiver(xs[::s,::s],ys[::s,::s],bx[::s,::s].T,by[::s,::s].T,color='white',scale=bmax*scale,
                  pivot='mid',width=0.013,headaxislength=5,headwidth=4,minshaft=1.8,edgecolors='k',lw=.3)
        
    
    # Contour plot of |B|
    if B_contour==True:
        mag_range = 2*np.max(mag_B)
        n_levels = int(mag_range/mag_res)
        cs = ax.contour(mag_B.T,origin='lower',levels=n_levels, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
        
    # Contour plot of phase
    if type(phase)!=type(None):
        phase_range = (np.max(phase)-np.min(phase))/1e-9
        n_levels = int(phase_range/phase_res)
        cs = ax.contour(phase.T-np.min(phase).T,origin='lower',levels=10, extent=[p1[0], p2[0], p1[1],p2[1]], alpha = .3,colors='white')
    
    ax.axis('off')
    
def save_B_to_paraview(fpath,bx,by,bz):
    """ Export B arrays to .vts paraview file
      
    Once in paraview:
    * apply python calculator, in expression put 'make_vector(u,v,w)'
    * extract subset to desired size
    * for streamtracer, apply filter or press button. in colorbar select enable opacity to hide small magnitude. use pointcloud seeds for better result
    * for arrows, apply glyph filter, make sure orientation is 'result'

    """
    
    dim = bx.shape
    x = np.arange(dim[0])
    y = np.arange(dim[1])
    z = np.arange(dim[2])
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    
    gridToVTK(fpath,X,Y,Z,pointData={'u':np.ascontiguousarray(bx),'v':np.ascontiguousarray(by),'w':np.ascontiguousarray(bz)})
    
def NRMSE(A,B):
    """ A = ground truth, B = reconstruction"""
    prefactor = 1/(np.max(A)-np.min(A))
    abs_diff = np.sum((A-B)**2)
    N = np.shape(A)[0]*np.shape(A)[1]*np.shape(A)[2]
    nrmse = prefactor * (1/N*abs_diff)**.5
    return nrmse

def CC(A,B):
    """ A = ground truth, B = reconstruction"""
    N = np.shape(A)[0]*np.shape(A)[1]*np.shape(A)[2]
    num = N*np.sum(A*B) - np.sum(A)*np.sum(B)
    denom = ((N*np.sum(A**2)-np.sum(A)**2)*(N*np.sum(B**2)-np.sum(B)**2))**0.5
    return (num/denom)

def MAAPE(A,B):
    """ A = ground truth, B = reconstruction, 0 good, 1 bad"""
    N = np.shape(A)[0]*np.shape(A)[1]*np.shape(A)[2]
    maape = 1/N*np.sum(np.arctan2(abs((A-B)),abs(A)))
    return maape

def test_metric(A,B,fun):
    print('Actual \t',fun(A,A))
    print('Data \t',fun(A,B))
    print('Random 1e-10 \t',fun(A,np.zeros_like(B)-(1e-10)*(0.05+0.1*np.random.rand(np.shape(B)[0],np.shape(B)[1],np.shape(B)[2]))))
    print('Random 1e0 \t',fun(A,np.ones_like(B)-0.05+0.1*np.random.rand(np.shape(B)[0],np.shape(B)[1],np.shape(B)[2])))
    print('Random 1e10\t',fun(A,1e10*(0.5-np.ones_like(B)*np.random.rand(np.shape(B)[0],np.shape(B)[1],np.shape(B)[2]))))
    
def check_memory(g,d):
    """ Return current RAM use in gigabytes for:
    - your current python instance (total)
    - The objects in your current python environment
    
    pass in globals() and dir() so it works
    """
    import os
    import psutil
    import inspect
    print('--- RAM usage / GB ---')
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
    print('Total:\t %.3f' % memoryUse)
    
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    mems = sorted([(x, sys.getsizeof(g.get(x))) for x in d], key=lambda x: x[1], reverse=True)
    vals = np.sum(np.array(mems)[:,1].astype('float32'))/1e9
    print('Local:\t %.3f' %vals)
    
def plot_B_series(bx,by,bz,slices=None):
    fig,axs=plt.subplots(ncols=len(slices),figsize=(15,3))
    for i,i_slice in enumerate(slices):
        if i == len(slices)-1:
            plot_3d_B_slice(bx,by,bz,ax=axs[i],i_slice=i_slice,cbar=True)
        else:
            plot_3d_B_slice(bx,by,bz,ax=axs[i],i_slice=i_slice)
        
    plt.tight_layout()
    
def generate_A_projection(AX,AY,AZ,angles,mesh_params=None,unpad=False,reorient = False):
    """ Returns A projections for given angles
    in order [x, i_tilt, y] """
    # Initialise parameters
    ax_projs = []
    ay_projs = []
    az_projs = []
    if mesh_params == None:
        p1 = (0,0,0)
        s = np.shape(MX)
        p2 = (s[0],s[1],s[2])
        n = p2
        mesh_params = [p1,p2,n]
    
    # Loop through projection angles
    for i in range(len(angles)):
        ax,ay,az = angles[i]
        #rotate A
        AXr = rotate_bulk(AX,ax,ay,az)
        AYr = rotate_bulk(AY,ax,ay,az)
        AZr = rotate_bulk(AZ,ax,ay,az)
        ax_proj = project_along_z(AXr,mesh_params=mesh_params)
        ay_proj = project_along_z(AYr,mesh_params=mesh_params)
        az_proj = project_along_z(AZr,mesh_params=mesh_params)
        
        # reorient to match phase_projs
        if reorient == True:
            ax_proj = np.flipud(ax_proj.T)
            ay_proj = np.flipud(ay_proj.T)
            az_proj = np.flipud(az_proj.T)
        
        
        #calculate phase
        ax_projs.append(ax_proj)
        ay_projs.append(ay_proj)
        az_projs.append(az_proj)
    
    # Prepare projections for reconstruction
    ax_projs = np.transpose(ax_projs,axes=[1,0,2]) # reshape so proj is middle column
    ax_projs=ax_projs.astype(np.float32)
    ay_projs = np.transpose(ay_projs,axes=[1,0,2]) # reshape so proj is middle column
    ay_projs=ay_projs.astype(np.float32)
    az_projs = np.transpose(az_projs,axes=[1,0,2]) # reshape so proj is middle column
    az_projs=az_projs.astype(np.float32)
    
    return np.array(ax_projs),np.array(ay_projs),np.array(az_projs)


def calculate_A_contributions(angles):
    """ For a given tilt angle series [[-70,0,0], [-60,0,0],...]
    calculate the weighting that the x,y,z components of A will contribute
    to each phase image in the series """
    
    ws = []
    
    for i, a in enumerate(angles):
        # calculate rotation matrix
        mrot = rotation_matrix(a[0],a[1],a[2])

        # Calculate position of x,y,z axes after rotation
        nx = np.dot(mrot,[1,0,0])
        ny = np.dot(mrot,[0,1,0])
        nz = np.dot(mrot,[0,0,1])
        
        #print(nx)

        # calculate how aligned the new x,y,z axes are with the beam direction
        # i.e. how much does this component contribute to the phase image?
        nx = np.dot(nx,[0,0,1])
        ny = np.dot(ny,[0,0,1])
        nz = np.dot(nz,[0,0,1])
        
        ws.append([nx,ny,nz])
    return np.array(ws)

def weight_phases(projs,ws):
    """ For a specific projection component, and its set of weights,
    multiplies those weights through the projection data 
    # checked and this definitely returns new_ps in same orientation as phase_proj """
    new_ps = []
    for i,w in enumerate(ws):
        p = projs[:,i,:]
        new_p = p*w
        new_ps.append(new_p)
        
    new_ps = np.transpose(new_ps,axes=[1,0,2]) # reshape so proj is middle column
    
    return new_ps

def update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws):
    """ Given a set of phase projection data, and the current best guess for each component af A,
        returns a new set projection data for each component, which is the raw data after removing
        the contribution of the other 2 components and reweighting it back to unity """
    const = -np.pi/constants.codata.value('mag. flux quantum')/(2*np.pi)
    new_x = phase_projs*1/const - weight_phases(a_weighted_y,ws[:,1]) - weight_phases(a_weighted_z,ws[:,2])
    new_x = weight_phases(new_x,1/ws[:,0])

    new_y = phase_projs*1/const - weight_phases(a_weighted_x,ws[:,0]) - weight_phases(a_weighted_z,ws[:,2])
    new_y = weight_phases(new_y,1/ws[:,1])

    new_z = phase_projs*1/const - weight_phases(a_weighted_y,ws[:,1]) - weight_phases(a_weighted_x,ws[:,0])
    new_z = weight_phases(new_z,1/ws[:,2])
    
    return new_x,new_y,new_z

def recon_step(a_projs,ws,angles,mesh_params, thresh=0.706, algorithm = 'SIRT3D_CUDA', niter=40, weight = 0.001,
                            balance = 1, steps = 'backtrack', callback_freq = 0):
    """ Given a set of A-component projections along with their weights and angles, does a SIRT reconstruction on it.
    It will only use projections where the component accounted for >threshold % of the original data in that slice.
    
    Input: Tilt series for a component of A, with it's associated weightings, angles and mesh parameters
    Specify: The threshold for this roudn of nmaj (thresh) and the number of iterations (niter)
    Return: A 3D reconstruction of 1 component of A """
    
    # Initialise parameters
    p1,p2,nn=mesh_params
    res=p2[0]/nn[0]
    a_thresh = []
    angles_thresh = []
    
    # Threshold out data with low weighting
    for i,w in enumerate(ws):
        if abs(w) > thresh:
            a_thresh.append(a_projs[:,i,:])
            angles_thresh.append(angles[i])
            
    angles_thresh=np.array(angles_thresh)
    a_thresh = np.transpose(a_thresh,axes=[1,0,2]) # reshape so proj is middle column  

    # Perform SIRT reconstruction on remaining data
    vecs = generate_vectors(angles_thresh)
    recon = generate_reconstruction(a_thresh,vecs, algorithm = algorithm, niter=niter, weight = weight,
                                balance = balance, steps = steps, callback_freq = callback_freq)
    
    # reformat to match structure of input data  
    recon = np.transpose(recon,axes=(2,1,0))[:,::-1,:]
    
    # Rescale intensities to account for pixel size
    recon = recon/res
    
    # Ensure astra doesn't fill up the RAM
    astra.clear()
    
    return recon

def iterative_update_algorithm(phase_projs,angles,mesh_params,n_pad,n_full_iter=1,n_step_iter=5, 
                               algorithm = 'SIRT3D_CUDA', weight = 0.001,thresh_range=(.01,.7),callback=False):
    """ Puts everything together for the multi-axis reconstruction procedure 
    Input: Phase tilt series, associated angles, mesh parameters, and pad count in pixels
    Specify: nmaj (n_full_iter), nmin (n_step_iter), and threshold range (tmin,tmax)
    Returns: Reconstructed Ax, Ay, Az arrays """
    
    if callback == True:
        callback_freq = 1
    else:
        callback_freq = 0
    
    # Calculate weightings for each tilt angle
    ws = calculate_A_contributions(angles)
    
    # In default run, threshold will initially be high (tmax)
    tmin,tmax = thresh_range
    # generate linearly spaced threshold list from low to high threshold for each nmaj
    possible_ts = np.linspace(tmin,tmax,n_full_iter-1)
    thresh = tmax
    
    # initialize new arrays
    a_weighted_x = np.zeros_like(phase_projs)
    a_weighted_y = np.zeros_like(phase_projs)
    a_weighted_z = np.zeros_like(phase_projs)
    
    # Generate A(0) tilt series
    a_weighted_x, a_weighted_y, a_weighted_z = update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws)
    
    if callback == True:
        print("Initialised")
      
    # do first step of reconstruction
    Ax_recon = recon_step(a_weighted_x,ws[:,0],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    Ay_recon = recon_step(a_weighted_y,ws[:,1],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    Az_recon = recon_step(a_weighted_z,ws[:,2],angles,mesh_params,niter=n_step_iter,algorithm=algorithm,weight=weight,thresh=thresh, callback_freq =callback_freq) 
    
    if callback == True:
        print("Iteration 1 finished")
    
    # Cycle through iterations for nmaj>1
    for i in range(n_full_iter-1):
        
        # Repeat t=tmax for nmaj=2, then decrease t for subsequent iterations
        thresh = possible_ts[-(i+1)]
        # recalculate projection data
       
        # project current A to get A_p(n)
        n=n_pad
        a_weighted_x,a_weighted_y,a_weighted_z = generate_A_projection_fast(Ax_recon[n:-n,n:-n,n:-n],Ay_recon[n:-n,n:-n,n:-n],Az_recon[n:-n,n:-n,n:-n],angles,mesh_params=mesh_params,reorient=True)
        
        # Update to get A_p(n+1)
        a_weighted_x = np.pad(a_weighted_x,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_y = np.pad(a_weighted_y,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_z = np.pad(a_weighted_z,[(n_pad,n_pad),(0,0),(n_pad,n_pad)], mode='constant', constant_values=0)
        a_weighted_x, a_weighted_y, a_weighted_z = update_weighted_proj_data(phase_projs,a_weighted_x,a_weighted_y,a_weighted_z,ws)
        
        # SIRT reconstruct to get A(n+1)
        Ax_recon = recon_step(a_weighted_x,ws[:,0],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        Ay_recon = recon_step(a_weighted_y,ws[:,1],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        Az_recon = recon_step(a_weighted_z,ws[:,2],angles,mesh_params,niter=n_step_iter,thresh=thresh,algorithm=algorithm,weight=weight, callback_freq =callback_freq) 
        
        # ensure astra doesn't clog up the RAM
        astra.clear()
        
        if callback == True:
            print("Iteration ",i+2," finished")
    
    return Ax_recon,Ay_recon,Az_recon

def plot_component_orthoslices(X,Y,Z, vmin=False, vmax=False,npad=False, i = None, oslice = 'z'):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    f,axs = plt.subplots(ncols=3,figsize=(14,4))
    
    # Check whether to set intensity limits to maximal extent
    if vmin == True:
        vmin = np.min([X,Y,Z])
        vmax = np.max([X,Y,Z])
            
    if vmin == False:
        vmin=None
        vmax = None
            
    # Choose what to plot
    if npad != False: # unpad
        x = X[npad:-npad,npad:-npad,npad:-npad]
        y = Y[npad:-npad,npad:-npad,npad:-npad]
        z = Z[npad:-npad,npad:-npad,npad:-npad]
    if i == None: # find central slice
        i = int(np.shape(x)[0]/2)
    if oslice=='z':
        x = x[:,:,i]
        y = y[:,:,i]
        z = z[:,:,i]
    if oslice=='y':
        x = x[:,i,:]
        y = y[:,i,:]
        z = z[:,i,:]
    if oslice=='x':
        x = x[i,:,:]
        y = y[i,:,:]
        z = z[i,:,:]
    
    # Plot X
    im1 = axs[0].imshow(x,vmin=vmin,vmax=vmax)
    
    if vmin == None:
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im1, cax=cax, orientation='vertical')
            
    # Plot Y
    im2 = axs[1].imshow(y,vmin=vmin,vmax=vmax)
    
    if vmin == None:
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im2, cax=cax, orientation='vertical')
            
    # Plot Z
    im3 = axs[2].imshow(z,vmin=vmin,vmax=vmax)
    
    divider = make_axes_locatable(axs[2]) # switch on colorbar
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar(im3, cax=cax, orientation='vertical')
    
    for ax in axs:
        ax.axis('off')
        
    f.patch.set_facecolor('white')
    
    plt.tight_layout()
    
def omf_to_mag(data):
    """ Extract magnetization in grid array from ubermag 'system' object """
    Mx = data[:,:,:,0]
    My = data[:,:,:,1]
    Mz = data[:,:,:,2]
    #ms = system.m.array
#     ms = data
#     shape = np.shape(ms)

#     xs,ys,zs,mx,my,mz = [],[],[],[],[],[]
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             for k in range(shape[2]):
#                 xs.append(i)
#                 ys.append(j)
#                 zs.append(k)
#                 mx.append(ms[i][j][k][0])
#                 my.append(ms[i][j][k][1])
#                 mz.append(ms[i][j][k][2])
#     Mx,My,Mz = np.reshape(mx,(shape[0],shape[1],shape[2])),\
#                             np.reshape(my,(shape[0],shape[1],shape[2])), \
#                             np.reshape(mz,(shape[0],shape[1],shape[2]))
    return Mx,My,Mz

def spatial_freq_filter(ps, rad=10):
    """ Takes a stack of images (indexed in middle column)
    and applys a circular cutoff in Fourier space at a radius
    defined in pixels by rad """
    
    num = np.shape(ps)[1]
    
    ps_filt = []
    for ipic in range(num):
        im = ps[:,ipic,:]
        ft = np.fft.fftshift(np.fft.fft2(im))

        mask = np.zeros_like(ft,dtype='uint8')
        cent = np.shape(mask)[0]/2
        for i, mi in enumerate(mask):
            for j, mij in enumerate(mi):
                x = cent - i
                y = cent - j
                if x**2 + y**2 < rad**2:
                    mask[i,j] = 1

        ftfilt = mask*ft
        imfilt = np.real(np.fft.ifft2(np.fft.fftshift(ftfilt)))
        
        ps_filt.append(imfilt)
        
    ps_filt = np.transpose(ps_filt,axes=[1,0,2]) # reshape so proj is middle column
    ps_filt=ps_filt.astype(np.float32)

    return ps_filt

def misalign_func(ps,maxshift=1):
    """ Takes a stack of images (indexed along middle column)
    and randomly translates them up/down and left/right by 
    up to a maximum of maxshift pixels """
    
    num = np.shape(ps)[1]
    
    ps_shift = []
    for ipic in range(num):
        im = ps[:,ipic,:]
        imshift = copy.deepcopy(im)

        xtrans = np.random.randint(-1,high=2)
        if xtrans!=0:
            if xtrans == 1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[val:]
                imshift = np.pad(imshift,((0,val),(0,0)),mode='edge')
                #print('right',val)
            if xtrans == -1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:-val]
                imshift = np.pad(imshift,((val,0),(0,0)),mode='edge')
                #print('left',val)

        ytrans = np.random.randint(-1,high=2)       
        if ytrans!=0:
            if ytrans == 1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:,val:]
                imshift = np.pad(imshift,((0,0),(0,val)),mode='edge')
                #print('up',val)
            if ytrans == -1:

                val = np.random.choice(list(range(1,maxshift+1,1)))
                imshift = imshift[:,:-val]
                imshift = np.pad(imshift,((0,0),(val,0)),mode='edge')
                #print('down',val)
                
        ps_shift.append(imshift)
    
    ps_shift = np.transpose(ps_shift,axes=[1,0,2]) # reshape so proj is middle column
    ps_shift=ps_shift.astype(np.float32)
    
    return ps_shift

def plot_phases_interactive(phis,angles=None):
    """ Plots phase shifts in an interactive way 
    
    Note that even though it is not an import, you must have installed:
    jupyterlab_widgets (and potentially ipympl) and probably then update ipywidgets too
    Must then restart jupyterlab for this to work properly.
    """
    lim = np.shape(phis)[1]-1

    def update(i):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(phis[:,int(i),:],cmap='Greys_r',vmin=np.min(phis),vmax=np.max(phis))
        cbar = plt.colorbar(im,fraction=0.046, pad=0.04)
        cbar.ax.set_title('Phase shift / rads', rotation=0,fontsize=14)
        vals = [100, 30,10]
        if type(angles) != type(None):
            title = r'($%.1f^{\circ},%.1f^{\circ},%.1f^{\circ})$' % tuple(angles[int(i)].tolist())
            plt.title(title,fontsize=13)
        for val in vals:
            if np.pi/val < np.max(phis):
                cbar.ax.plot([-1,1], [np.pi/val,np.pi/val],'r-',markersize=50)
                cbar.ax.text(1.3,np.pi/val,r'$\pi$ / %.i' % val,fontsize=13,color='r')
                ax.axis('off')

    ipywidgets.interact(update,i=(0,lim,1))
    
def noisy_phase(ps, misalign = False, gaussian = False, lowpass=False,maxshift=3,noise_level=np.pi/30,freq_rad_px=20,holo=False,fringe=10,up=10,v=1,n=None,c=1000,n_pad=32,MX=None,MY=None,MZ=None,angles=None,mesh_params=None,fxc=400,fyc=400,rc=50):
    """ Makes phase images noisy!
    Misalign: Randomly shifts each image in stack by +- maxshift pixels in x and y directions.
    Gaussian: Adds Gaussian noise where noise_level corresponds to three standard deviations.
    Lowpass: Removes frequencies beyond a freq_rad_px radius in Fourier space.
    """
    ps_n = ps
    if misalign == True:
        # Randomly shift the tilt series
        ps_n = misalign_func(ps_n,maxshift=maxshift)

    if gaussian == True:
        # Add gaussian noise
        ps_n = noisy(ps_n,noise_typ='gauss',g_var=(noise_level/3)**(2))

    if lowpass == True:
        # Filter out high spatial frequencies
        ps_n = spatial_freq_filter(ps_n,rad=freq_rad_px)
        
    if holo==True:
        ps_n = hologram_noise(ps_n,MX,MY,MZ,mesh_params,angles,fxc=fxc,fyc=fyc,rc=rc,n=n,v=v,c=c,plot=False,fringe=fringe)
    
    return ps_n

def create_circular_mask(imsize, center=(None), radius=None):
    h,w = imsize,imsize
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    
    mask = np.ones((h,w))
    mask[dist_from_center >= radius] = 0
    #mask = dist_from_center <= radius
    
    # blur edge of mask
    mask = abs(ndimage.gaussian_filter(mask, sigma=1))
    
    return mask



def apply_circular_mask(circ_mask,im_ft):
    masked_im = im_ft.copy()
    masked_im = masked_im*circ_mask
    #masked_im[~circ_mask] = 0
    return masked_im

def centre_sideband(masked_im,fxc,fyc,rc):
    """ Centre by cropping around sideband centre then padding back out with zeros """
    size = np.shape(masked_im)[0]
    cropped_im = masked_im[fyc-2*rc:fyc+2*rc,fxc-2*rc:fxc+2*rc]
    size_cropped = np.shape(cropped_im)[0]
    size_pad = size-size_cropped
    centred_im = np.pad(cropped_im,int(size_pad/2),mode='constant')
    return centred_im

def extract_wf_and_phase(ob,ref):

    extracted_ob = np.fft.ifft2(ob)
    extracted_ref = np.fft.ifft2(ref)

    psi_0 = extracted_ob / extracted_ref
    phase = np.arctan2(np.imag(psi_0),np.real(psi_0)) # or np.angle(psi_0)

    return psi_0, phase

def reconstruct_hologram(holo,ref,fxc=135, fyc = 125,rc = 8,plot=False):
    ## hamming filter
    # create window
    size = np.shape(holo)[0]
    ham1d = np.hamming(size)
    ham2d = (np.outer(ham1d,ham1d))**1 # 0.5 normalises but might not remove cross

    # apply window
    ob = ham2d*holo
    ref = ham2d*ref
    
    ## FT
    ob_ft = np.fft.fft2(ob)
    ob_ft = np.fft.fftshift(ob_ft)
    ref_ft = np.fft.fft2(ref)
    ref_ft = np.fft.fftshift(ref_ft)
    
    ## select sideband
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    
    # auto-find centre
    maxval = np.amax(masked_ob)
    maxind = np.where(masked_ob == maxval)
    fxc,fyc = maxind[1][0], maxind[0][0]


    # auto-find radius
    dx = size/2 - fxc
    dy = size/2 - fyc
    rc = (dx**2 + dy**2)**0.5 / 2 #- 10

    # remask
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    

    if plot==True:
        plt.imshow(np.log10(abs(ob_ft)))
        plt.imshow(np.log10(abs(masked_ob)),cmap='Blues_r')
        plt.show()

    centred_ob = centre_sideband(masked_ob,fxc,fyc,int(rc))
    centred_ref = centre_sideband(masked_ref,fxc,fyc,int(rc))
    
    ## reconstruct wave function
    psi_0, phase = extract_wf_and_phase(centred_ob,centred_ref)
    
    ## unwrap phase
    pu = unwrap_phase(phase)
    #pu = (pu+abs(np.min(pu))) 
    
    return pu
    

def hologram_noise(ps,MX,MY,MZ,mesh_params,angles,n_pad=30,v=1,n=None,c=1000,fxc=400,fyc=400,rc=50,plot=False,fringe=10,up=10):
    """ Add realistic hologram-type noise to a phase image 
    
    Fringe spacing - Needs to be min 4 pix per fringe, avg camera is 2k. Input dim of 40x40, upscale by 10 so 400x400
                     if 4 for 2k = 4 for 400, should become
    
    Direct from libertem:
        holo = counts / 2 * (1. + amp ** 2 + 2. * amp * visibility
                             * np.cos(2. * np.pi * y / sampling * np.cos(f_angle)
                                      + 2. * np.pi * x / sampling * np.sin(f_angle)
                                      - phi))

        noise_scale = poisson_noise * counts
        holo = noise_scale * np.random.poisson(holo / noise_scale)
    
    """
    num = np.shape(ps)[1]
    
    ps_n = []
    
    for ipic in range(num):
        im = ps[:,ipic,:]
        phase_m = copy.deepcopy(im)
    
        # Upsample
        phase_tot = zoom(phase_m,(up,up))

        # estimate amplitude
        a=angles[ipic]
        MXr,MYr,MZr=rotate_magnetisation(MX,MY,MZ,a[0],a[1],a[2])
        mag =(MXr**2+MYr**2+MZr**2)**.5 # magnitude of magnetisation
        mag = mag/np.max(mag) # rescale so 1 is max
        thickness = project_along_z(mag,mesh_params=mesh_params) # project
        thickness = np.pad(thickness,[(n_pad,n_pad),(n_pad,n_pad)],mode='edge')
        amp = 1-thickness/np.max(thickness)/2
        amp = zoom(amp,(up,up))

        # Create hologram and reference
        holo = hologram_frame(np.ones_like(phase_tot), phase_tot,sampling=fringe,visibility=v,poisson_noise=n,counts=c)
        ref = hologram_frame(np.ones_like(phase_tot), np.zeros_like(phase_tot),sampling=fringe,visibility=v,poisson_noise=n)

        # Extract phase
        pu = reconstruct_hologram(holo,ref,fxc=fxc,fyc=fyc,rc=rc,plot=False)

        # Downsample
        phase_recon = zoom(pu,(1/up,1/up))
        
        ps_n.append(phase_recon)
        
    ps_n = np.transpose(ps_n,axes=[1,0,2]) # reshape so proj is middle column
    ps_n=ps_n.astype(np.float32)
    
    return ps_n

def check_holo_params(im,MX,MY,MZ,mesh_params,a=(0,0,0),n_pad=30,fxc=400,fyc=400,rc=50,up=10,fringe=15,v=1,n=None,c=1000):
    phase_tot = zoom(im,(up,up))
    
    MXr,MYr,MZr=rotate_magnetisation(MX,MY,MZ,a[0],a[1],a[2])
    mag =(MXr**2+MYr**2+MZr**2)**.5 # magnitude of magnetisation
    mag = mag/np.max(mag) # rescale so 1 is max
    thickness = project_along_z(mag,mesh_params=mesh_params) # project
    thickness = np.pad(thickness,[(n_pad,n_pad),(n_pad,n_pad)],mode='edge')
    amp = 1-thickness/np.max(thickness)/2
    amp = zoom(amp,(up,up))
    
    # Create hologram and reference
    holo = hologram_frame(amp, phase_tot,sampling=fringe,visibility=v,poisson_noise=n,counts=c)
    ref_hol = hologram_frame(np.ones_like(phase_tot), np.zeros_like(phase_tot),sampling=fringe,visibility=v,poisson_noise=n)
    
    size = np.shape(holo)[0]
    ham1d = np.hamming(size)
    ham2d = (np.outer(ham1d,ham1d))**1 # 0.5 normalises but might not remove cross

    # apply window
    ob = ham2d*holo
    ref = ham2d*ref_hol
    
    ## FT
    ob_ft = np.fft.fft2(ob)
    ob_ft = np.fft.fftshift(ob_ft)
    ref_ft = np.fft.fft2(ref)
    ref_ft = np.fft.fftshift(ref_ft)
    
    # auto-find radius
    
    if rc == None:
        n_sample=2
        n_pad=32
        dx = (size-n_sample*n_pad)/2 - fxc
        dy = (size-n_sample*n_pad)/2 - fyc
        # dx = dx-.5*n_pad
        # dy = dy-.5*n_pad
        rc = (dx**2 + dy**2)**0.5 / 2 #- 10
        print(rc)
    
    ## select sideband
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    
    # auto-find centre
    maxval = np.amax(masked_ob)
    maxind = np.where(masked_ob == maxval)
    fxc,fyc = maxind[1][0], maxind[0][0]

    # remask
    circ_mask = create_circular_mask(size,center=[fxc,fyc],radius=rc)
    masked_ob = apply_circular_mask(circ_mask,ob_ft)
    masked_ref = apply_circular_mask(circ_mask,ref_ft)
    

    #
    
    f,axs = plt.subplots(ncols=3,nrows=2,figsize=(12,8))
    axs[0,0].imshow(holo,cmap='gray')
    axs[0,1].imshow(ref_hol,cmap='gray')
    axs[0,2].imshow(np.log10(abs(ob_ft)))
    axs[0,2].imshow(np.log10(abs(masked_ob)),cmap='Blues_r')
    
    axs[1,0].imshow(im)
    
    ps = [im,im]
    ps = np.transpose(ps,axes=[1,0,2])
    
    recon = hologram_noise(ps,MX,MY,MZ,mesh_params,[a,a],fxc=fxc,fyc=fyc,rc=rc,n=n,v=v,c=c,plot=False,fringe=fringe)
    recon = recon[:,0,:]
    print(fxc,fyc,rc)
    axs[1,1].imshow(recon)
    
    diff = axs[1,2].imshow(abs(im-recon),cmap='hot')
    plt.colorbar(mappable=diff,fraction=0.046, pad=0.04)
    plt.tight_layout()
    #plt.subplots_adjust(hspace=-1)
    
    axs[0,0].set_title('Object hologram',fontsize=14)
    axs[0,1].set_title('Reference hologram',fontsize=14)
    axs[0,2].set_title('Selected sideband',fontsize=14)
    axs[1,0].set_title('Input phase',fontsize=14)
    axs[1,1].set_title('Output phase',fontsize=14)
    axs[1,2].set_title('Absolute difference',fontsize=14)
    
def generate_A_projection_fast(AX,AY,AZ,angles,mesh_params=None,unpad=False,reorient = True):
    """ Returns A projections for given angles
    in order [x, i_tilt, y], using astra forward projector """
    
    # Define some astra-specific things
    # Fairly sure that these ones in mm don't matter for parallel geom
    distance_source_origin = 300  # [mm]
    distance_origin_detector = 100  # [mm]
    detector_pixel_size = 1.05  # [mm]
    detector_rows = AX.shape[0]  # Vertical size of detector [pixels].
    detector_cols = AX.shape[0]  # Horizontal size of detector [pixels].
    
    # Create astra vectors to describe angles
    vecs = generate_vectors(angles)

    # AX
        # Create volume geometry
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    
        # Reorient to match with old version
    AX = np.transpose(AX,[2,1,0])
    AX = AX[:,::-1,:]
    
        # Load data into astra
    phantom_idx = astra.data3d.create('-vol', vol_geom, data=AX)
    
        # Create projection geometry
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
        
        # Get forward projections
    projections_idx, projectionsx = astra.creators.create_sino3d_gpu(phantom_idx, proj_geom, vol_geom)
    
        # Clear astra memory
    astra.clear()
    
    # AY
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    AY = np.transpose(AY,[2,1,0])
    AY = AY[:,::-1,:]
    phantom_idy = astra.data3d.create('-vol', vol_geom, data=AY)
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
    projections_idy, projectionsy = astra.creators.create_sino3d_gpu(phantom_idy, proj_geom, vol_geom)
    astra.clear()
    
    # AZ
    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                              detector_rows)
    AZ = np.transpose(AZ,[2,1,0])
    AZ = AZ[:,::-1,:]
    phantom_idz = astra.data3d.create('-vol', vol_geom, data=AZ)
    proj_geom = astra.create_proj_geom('parallel3d_vec', detector_rows, detector_cols, 
                                       np.array(vecs),(distance_source_origin + distance_origin_detector) /detector_pixel_size, 0)
    projections_idz, projectionsz = astra.creators.create_sino3d_gpu(phantom_idz, proj_geom, vol_geom)
    astra.clear()
    
    # Scale correctly in line with previous version
    ax_projs = projectionsx*mesh_params[1][0]/mesh_params[2][0]
    ay_projs = projectionsy*mesh_params[1][0]/mesh_params[2][0]
    az_projs = projectionsz*mesh_params[1][0]/mesh_params[2][0]
    
    return np.array(ax_projs),np.array(ay_projs),np.array(az_projs)