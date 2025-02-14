
def get_f(path,filename):
    edge=pd.read_csv(path+'/'+filename+'_Edges.csv')
    node_t=pd.read_csv(path+'/'+filename+'_Feats_T.csv')
    node_s=pd.read_csv(path+'/'+filename+'_Feats_S.csv')
    node_i=pd.read_csv(path+'/'+filename+'_Feats_I.csv')
    node_t[['x', 'y']] = node_t['Centroid'].str.strip('[]').str.split(',', expand=True)
    node_t['x'] = node_t['x'].astype(float)
    node_t['y'] = node_t['y'].astype(float)
    node_s[['x', 'y']] = node_s['Centroid'].str.strip('[]').str.split(',', expand=True)
    node_s['x'] = node_s['x'].astype(float)
    node_s['y'] = node_s['y'].astype(float)
    node_i[['x', 'y']] = node_i['Centroid'].str.strip('[]').str.split(',', expand=True)
    node_i['x'] = node_i['x'].astype(float)
    node_i['y'] = node_i['y'].astype(float)
    return edge,node_t,node_s,node_i
def get_node(node_t,node_s,node_i,box):
    filtered_t = node_t[(node_t['x'] >= box[0][0]) & (node_t['x'] <= box[1][0]) & (node_t['y'] >= box[0][1]) & (node_t['y'] <= box[1][1])]
    filtered_s = node_s[(node_s['x'] >= box[0][0]) & (node_s['x'] <= box[1][0]) & (node_s['y'] >= box[0][1]) & (node_s['y'] <= box[1][1])]
    filtered_i = node_i[(node_i['x'] >= box[0][0]) & (node_i['x'] <= box[1][0]) & (node_i['y'] >= box[0][1]) & (node_i['y'] <= box[1][1])]
    return filtered_t,filtered_s,filtered_i
def get_edge(edge,s_name,t_name,s_map,t_map,tr):
    if tr==0:
        filtered_edge=edge[(edge['source'].isin(s_name)) & (edge['target'].isin(t_name))]
    else:
        filtered_edge1 = edge[(edge['source'].isin(s_name)) & (edge['target'].isin(t_name))]
        filtered_edge2 = edge[(edge['source'].isin(t_name)) & (edge['target'].isin(s_name))]
        filtered_edge2.loc[:, ['source', 'target']] = filtered_edge2.loc[:, ['target', 'source']].values
        filtered_edge = pd.concat([filtered_edge1, filtered_edge2], ignore_index=True)

    filtered_edge_s = pd.merge(filtered_edge, s_map, left_on='source', right_on='id', how='left')
    s_id = filtered_edge_s['mapped_id'].values
    filtered_edge_t = pd.merge(filtered_edge, t_map, left_on='target', right_on='id', how='left')
    t_id = filtered_edge_t['mapped_id'].values

    return np.stack([s_id, t_id], axis=0)
def get_graph(edge,node_t,node_s,node_i,y,window_bbox):
    if window_bbox!=None:
        filtered_t,filtered_s,filtered_i=get_node(node_t,node_s,node_i,window_bbox)
    else:
        filtered_t,filtered_s,filtered_i=node_t,node_s,node_i

    map_t=pd.DataFrame({'id': list(filtered_t["name"]), 'mapped_id': list(range(len(filtered_t)))})
    map_s=pd.DataFrame({'id': list(filtered_s["name"]), 'mapped_id': list(range(len(filtered_s)))})
    map_i=pd.DataFrame({'id': list(filtered_i["name"]), 'mapped_id': list(range(len(filtered_i)))})

    graph = HeteroData()
    graph["T"].node_id = torch.tensor(list(filtered_t["name"]))
    graph["T"].mapped_id=torch.tensor(list(range(len(filtered_t))))
    graph["T"].feature=torch.tensor(filtered_t.iloc[:, 4:27].values, dtype=torch.float)
    graph["S"].node_id = torch.tensor(list(filtered_s["name"]))
    graph["S"].mapped_id = torch.tensor(list(range(len(filtered_s))))
    graph["S"].feature=torch.tensor(filtered_s.iloc[:, 4:27].values, dtype=torch.float)
    graph["I"].node_id = torch.tensor(list(filtered_i["name"]))
    graph["I"].mapped_id = torch.tensor(list(range(len(filtered_i))))
    graph["I"].feature=torch.tensor(filtered_i.iloc[:, 4:27].values, dtype=torch.float)

    graph["T", "T-T", "T"].edge_index = torch.tensor(get_edge(edge,list(filtered_t["name"]),list(filtered_t["name"]),map_t,map_t,0))
    graph["S", "S-S", "S"].edge_index = torch.tensor(get_edge(edge,list(filtered_s["name"]),list(filtered_s["name"]),map_s,map_s,0))
    graph["I", "I-I", "I"].edge_index = torch.tensor(get_edge(edge,list(filtered_i["name"]),list(filtered_i["name"]),map_i,map_i,0))
    graph["T", "T-S", "S"].edge_index = torch.tensor(get_edge(edge,list(filtered_t["name"]),list(filtered_s["name"]),map_t,map_s,1))
    graph["T", "T-I", "I"].edge_index = torch.tensor(get_edge(edge,list(filtered_t["name"]),list(filtered_i["name"]),map_t,map_i,1))
    graph["S", "S-I", "I"].edge_index = torch.tensor(get_edge(edge,list(filtered_s["name"]),list(filtered_i["name"]),map_s,map_i,1))

    graph = T.ToUndirected()(graph)

    #graph.y=torch.tensor(y)
    return graph

def get_dataset(df,feature_dir):
    dataset=[]
    for index, row in df.iterrows():
        slide_id = row['slide_id']
        label = row['label']
        feature_path=feature_dir+"/feature_"+str(label)+"/"+slide_id

        dataset_x = []
        window_bbox_path=glob.glob(feature_path+"/*")
        for i in window_bbox_path:
            edge,node_t,node_s,node_i=get_f(i,slide_id)
            # file_sp=i.split("/")[-1].split("-")
            # window_bbox=np.array([file_sp]).reshape((2, 2)).astype(int)
            dataset_x.append(get_graph(edge,node_t,node_s,node_i,label,window_bbox=None))
        dataset_y = torch.tensor(label)
        if len(dataset_x)!=0:
            dataset.append([dataset_x, dataset_y,feature_path])
    return dataset
train_df = pd.read_csv('*/train.csv')
test_df = pd.read_csv('*/val.csv')
feature_dir ="*/graph_feature"
dataset_train=get_dataset(train_df,feature_dir)
dataset_test=get_dataset(test_df,feature_dir)

batch_slides = 2
train_loader = DataLoader(dataset_train, batch_size=batch_slides, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batch_slides, shuffle=False)
