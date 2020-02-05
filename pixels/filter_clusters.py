import sys
import h5py
import argparse
import numpy as np
def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('FILE', help="Input .h5 dataset")
    parser.add_argument("--minToT", default=0, metavar='N', type=int, help="Minimum ToT Value")
    parser.add_argument("--maxToT", default=700, metavar='N', type=int, help="Maximum ToT Value")
    parser.add_argument("--minCl", default=0, metavar='N', type=int, help="Minimum Cluster Size")
    parser.add_argument("--maxCl", default=700, metavar='N', type=int, help="Maximum Cluster Size") 
    parser.add_argument("--save", help="saveLocation") 

    settings = parser.parse_args()

    return settings

def load_and_filter(doc, minToT,maxToT,minCl,maxCl,save):
    f = h5py.File(doc,'r')

    clusters = f['clusters']
    pixels = f['clusters'][:, 0, :]
    incidents = f['incidents']
    trajectories = f['trajectories']
    size = list()
    tot = list()
    for pixel in pixels:
        tot.append(np.sum(pixel))
        size.append(np.count_nonzero(pixel))

    filteredIndices = list()
    for i in range(0,len(tot)):
        if tot[i] >= minToT:
            if tot[i] <= maxToT:
                if size[i] >= minCl:
                     if size[i] <= maxCl:
                         filteredIndices.append(i)
    print('you filtered out '+str(len(filteredIndices))+'('+str(float(len(filteredIndices))/float(len(clusters))*100)+'%)'+' items')

    newClusters = clusters[filteredIndices]
    newIncidents = incidents[filteredIndices]
    create_new_file(save,newClusters,newIncidents,trajectories,filteredIndices)

def create_new_file(save_location,Xclusters,Xincidents,Xtrajectories,indices):
    f = h5py.File(save_location,'w')
    f.create_dataset("clusters",(np.shape(Xclusters)))
    f['clusters'][...] = Xclusters
    f.create_dataset("incidents",(np.shape(Xincidents)))
    f['incidents'][...] = Xincidents
    g = f.create_group("/trajectories")
    for i in range(0,len(indices)):
        g.create_dataset(str(indices[i]),data = Xtrajectories[str(indices[i])])

def main():
    config = parse_arguments()
    load_and_filter(config.FILE,config.minToT,config.maxToT,config.minCl,config.maxCl,config.save)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
