from google.cloud import bigquery
from sklearn.cluster import KMeans 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



class Preprocess:
    def extraction():
        print('Extraction starts')
        time=5500000000

#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=""
        client = bigquery.Client()
        QUERY = (
             """
        select time, machine_id, capacity.cpus, capacity.memory 
        from `google.com:google-cluster-data`.clusterdata_2019_a.machine_events
        where time=0
        """)

        df_machine = client.query(QUERY).to_dataframe()

        QUERY = (
             """
        select  *
        from `google.com:google-cluster-data`.clusterdata_2019_a.collection_events
        where (time between 1 and {}) and type=1
        order by time asc
        """).format(time)

        df_collection = client.query(QUERY).to_dataframe()
        id_list=tuple(df_collection['collection_id'])

        QUERY = """
        select  time, collection_id, resource_request.cpus as cpus, resource_request.memory as memory
        from `google.com:google-cluster-data`.clusterdata_2019_a.instance_events
        where (time between 1 and {}) and collection_id in {}
        order by time asc
        """.format(time,id_list)

        df_instance = client.query(QUERY).to_dataframe()
        mach_list=tuple(df_machine['machine_id'])
        col_list=tuple(df_collection['collection_id'])

        QUERY = """
        select collection_id, machine_id, cycles_per_instruction as CPI
        from `google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
        where (collection_id in {}) and (machine_id in {})
        and (cycles_per_instruction is not NULL)
        """.format(col_list,mach_list)
        df_cpi = client.query(QUERY).to_dataframe()

        df_instance.to_csv('./data/instance.csv',mode='w')
        df_cpi.to_csv('./data/cpi.csv', mode='w')
        df_machine.to_csv('./data/machine.csv',mode='w')
        df_collection.to_csv('./data/collection.csv')
        print('Extraction done')

    def preprocess():
        print('Preprocess starts')

        machine_num=12
        collection_num=5
        seed=26
        df_machine=pd.read_csv("./data/machine.csv")
        df_instance=pd.read_csv("./data/instance.csv")
        df_cpi=pd.read_csv("./data/cpi.csv")
        df_collection=pd.read_csv("./data/collection.csv")

        test_x=df_machine['cpus']
        test_y=df_machine['memory']  

        counts = df_machine.groupby(by=[df_machine['cpus'],df_machine['memory']]).size().to_frame('size').reset_index()

        test_list=[list(pair) for pair in zip(test_x, test_y)]
        set_list = []
        for i in test_list:
            if i not in set_list:
                set_list.append(i)
        for i,idx in zip(set_list,range(machine_num)):
            index=df_machine[(df_machine['cpus']==i[0])&(df_machine['memory']==i[1])].index
            df_machine.loc[index,'cluster']=idx+1


        df_collection_cluster=df_instance.copy()
        df_collection_cluster['avg_cpus']=df_instance.groupby(['collection_id'])['cpus'].transform('mean')
        df_collection_cluster['avg_memory']=df_instance.groupby(['collection_id'])['memory'].transform('mean')
        df_collection_cluster=df_collection_cluster[['collection_id','avg_cpus','avg_memory']].drop_duplicates().reset_index(drop=True)


        test_x=df_collection_cluster['avg_cpus']
        test_y=df_collection_cluster['avg_memory']   
        
        kmeans = KMeans(n_clusters=collection_num,random_state=seed)
        points = pd.DataFrame(test_x.values, test_y.values).reset_index(drop=False)
        points.columns = ["x", "y"]
        points.head()
        kmeans.fit(points)
        result_by_sklearn = points.copy()
        result_by_sklearn["cluster"] = kmeans.labels_+1
        result_by_sklearn.head()
        df_collection_cluster['cluster']=result_by_sklearn['cluster']

        for id_ in df_instance['collection_id']:
            df_instance.loc[df_instance['collection_id']==id_,'cluster']=int(df_collection_cluster.loc[df_collection_cluster['collection_id']==id_,'cluster'])
        for idx in range(collection_num):
            cpu=df_instance.loc[df_instance['cluster']==idx+1,'cpus'].mean(axis=0)
            memory=df_instance.loc[df_instance['cluster']==idx+1,'memory'].mean(axis=0)
            df_instance.loc[df_instance['cluster']==idx+1,'avg_cpus_request']=cpu
            df_instance.loc[df_instance['cluster']==idx+1,'avg_memory_request']=memory
        for id_ in df_collection['collection_id']:
            N=len(df_instance.loc[df_instance['collection_id']==id_])
            df_instance.loc[df_instance['collection_id']==id_,'N']=N

        df_collection=df_collection[['time','collection_id']]
        for id_ in df_collection['collection_id']:
            df_collection.loc[df_collection['collection_id']==id_,'cluster']=df_instance.loc[df_instance['collection_id']==id_,'cluster'].iloc[0]
            df_collection.loc[df_collection['collection_id']==id_,'avg_cpus_request']=df_instance.loc[df_instance['collection_id']==id_,'avg_cpus_request'].iloc[0]
            df_collection.loc[df_collection['collection_id']==id_,'avg_memory_request']=df_instance.loc[df_instance['collection_id']==id_,'avg_memory_request'].iloc[0]
            df_collection.loc[df_collection['collection_id']==id_,'N']=df_instance.loc[df_instance['collection_id']==id_,'N'].iloc[0]
        i=0
        df_cpi['collection_cluster']=0
        df_cpi['machine_cluster']=0
        for idx in df_cpi[['collection_id','machine_id']].values:
            col=df_collection.loc[df_collection['collection_id']==idx[0],'cluster'].values[0]
            df_cpi.loc[df_cpi.index[i],'collection_cluster']=col
            mach=df_machine.loc[df_machine['machine_id']==idx[1],'cluster'].values[0]
            df_cpi.loc[df_cpi.index[i],'machine_cluster']=mach
            i+=1


        df_cpi['1/CPI']=1/df_cpi['CPI']
        df_cpi_stat=df_cpi.groupby(['collection_cluster','machine_cluster'])['1/CPI'].mean().reset_index(name='1/cpi_mean')
        df_cpi_stat['1/cpi_variance']=df_cpi.groupby(['collection_cluster','machine_cluster'])['1/CPI'].var().reset_index(name='1/cpi_variance')['1/cpi_variance']

        ##save preprocessed data
        df_instance.to_csv('./data/pre_instance.csv',mode='w')
        df_cpi_stat.to_csv('./data/pre_cpi.csv', mode='w')
        df_machine.to_csv('./data/pre_machine.csv',mode='w')
        df_collection.to_csv('./data/pre_collection.csv',mode='w')
        
        ##plots for data analysis
        
        test_x=df_machine['cpus']
        test_y=df_machine['memory']  
        #Generate a list of unique points
        points=list(set(zip(test_x,test_y))) 
        #Generate a list of point counts
        count=[len([x for x,y in zip(test_x,test_y) if x==p[0] and y==p[1]]) for p in points]
        #Now for the plotting:
        plot_x=[i[0] for i in points]
        plot_y=[i[1] for i in points]
        count=np.array(count)
        cmap = mpl.cm.Reds(np.linspace(0,1,100))
        cmap = mpl.colors.ListedColormap(cmap[10:,:-1])
        plt.scatter(plot_x,plot_y,c=count,cmap=cmap)
        plt.xlabel('CPU capacity')
        plt.ylabel('Memory capacity')
        plt.colorbar()
        plt.savefig('./result/machine_cluster.png')
        plt.clf()
        sns.scatterplot(x="x", y="y", hue="cluster", data=result_by_sklearn, palette="Set2")
        plt.xlabel('CPU request size')
        plt.ylabel('Memory request size')
        plt.savefig('./result/collection_cluster.png')
        plt.clf()

        plt.hist(df_collection['time']*10**(-6),alpha=0.5,bins=100)
        plt.xlabel('Arrival time (sec)')
        plt.ylabel('Number of collections')
        plt.savefig('./result/Arrival.png')
        plt.clf()
        
        df=df_instance.groupby(by=[df_instance['collection_id']]).size().to_frame('size').reset_index()['size']
        sns.ecdfplot(data=df, complementary=True)
        plt.loglog(base=10)
        plt.xlabel('Number of instances')
        plt.ylabel('Complementary CDF')
        plt.savefig('./result/CCDF.png')
        plt.clf()
        print('Preprocess done')
        
        
        
