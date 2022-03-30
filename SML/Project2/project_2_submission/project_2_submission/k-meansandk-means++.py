from Precode import *
import numpy
data = np.load('AllSamples.npy')
import matplotlib.pyplot as plt
import matplotlib


def k_means(c,data,k):
    for i in range(11):
        #assigning clusters
        d=np.array([pow(data[:,0]-c[i,0])+pow(data[:,1]-c[i,1]) for i in range(len(c))]).T
        clusters=np.argmin(d,1).reshape(300,1)
        result=np.concatenate([data,clusters],axis=1)
        #calculting objective function ! 
        total_distance=0
        distance = result
        for i in range(len(c)):
            matrix_i=distance[distance[:,2]==i]
            d=np.sum(np.array(pow(matrix_i[:,0]-c[i,0])+pow(matrix_i[:,1]-c[i,1])))
            total_distance+=d
        print(total_distance) 
       
        #centers
        centers=np.zeros([k,2])
        i=0
        while(i<=k)
            centers[i]=np.mean(distance[distance[:,2]==i],0)[:2]
            i+=1
        c = centers
        print(c)
    return result



def k_means_plus(point,k,data):
    c=point.reshape(1,2)
    for _ in range(k-1):
        minimum=0
        d=np.mean(np.array([pow(data[:,0]-c[i,0])+pow(data[:,1]-c[i,1]) for i in range(len(c))]).T,1)
        minimum=int(np.argmax(d,0))
        c=np.concatenate([c,data[minimum,:].reshape(1,2)],0)
        data=np.delete(data,minimum,0)
    return c

def main_plus():#getting all the k values for elbow plotting
    k1,i_point1,k2,i_point2 = initial_S1('2434') # please replace 0111 with your last four digit of your ID
    print(k1)
    print(i_point1)
    print(k2)
    print(i_point2)
    Loss_=[]
    i=2
    while(i<=10)
        c=k_means_plus(i_point2,i,data)
        for _ in range(20):
            #assigning clusters
            d=np.array([pow(data[:,0]-c[i,0])+pow(data[:,1]-c[i,1]) for i in range(len(c))]).T
            clusters=np.argmin(d,1).reshape(300,1)
            result=np.concatenate([data,clusters],axis=1)
            total_distance=0
            distance = result
            for i in range(len(c)):
            matrix_i=distance[distance[:,2]==i]
            d=np.sum(np.array(pow(matrix_i[:,0]-c[i,0])+pow(matrix_i[:,1]-c[i,1])))
            total_distance+=d
            print(total_distance) 
            c=np.zeros([k,2])
            for i in range(k):
            c[i]=np.mean(aug_matrix[aug_matrix[:,2]==i],0)[:2]
            print(c)
        Loss_.append(loss_1)
        fig = plt.figure()
        i+=1
    plt.plot(range(2,10),Loss_[:])
    ax=plt.axes()
    ax.set_xlabel('number_of_cluster', fontsize=18)
    ax.set_ylabel('Loss', fontsize=16)
    ax.set_title('Loss vs. number_of_clusters in second strategy',fontsize=15)
    colors = ['yellow','blue','green','purple','pink','yellow']
    plt.scatter(result[:,0],result[:,1], c=result[:,2], cmap=matplotlib.colors.ListedColormap(colors))
    ax=plt.axes()
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_title('k-mean++')
    plt.show

def main():
    k1,i_point1,k2,i_point2 = initial_S1('2434') # please replace 0111 with your last four digit of your ID
    print(k1)
    print(i_point1)
    print(k2)
    print(i_point2)
    #for k = 3
    result = k_means(i_point1,data,k=4)
    colors = ['pink','purple','yellow','purple','orange']
    plt.scatter(result[:,0],result[:,1], c=result[:,2], cmap=matplotlib.colors.ListedColormap(colors))
    ax=plt.axes()
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_title('k=3',fontsize=14)
    #for k = 5
    output2 = k_means(i_point2,data,k=6)
    colors = ['pink','purple','yellow','green','orange']
    fig=plt.figure( dpi=80, figsize=(10,10), facecolor='w', edgecolor='k')

    plt.scatter(output2[:,0],output2[:,1], c=output2[:,2], cmap=matplotlib.colors.ListedColormap(colors),  s=200,
                marker='p')
    plt.show()
    ax=plt.axes()
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.set_title('k=5',fontsize=14)
    
    #--------kmeans-plus-plus---------------_#
    
    
main()
main_plus()
