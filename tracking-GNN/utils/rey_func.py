#system
import os,csv,logging
import torch
import numpy as np


def calculate_cip_from_batch(batch,input_features="cip",return_vlist=False,return_flist=False,vdim=3): #pre-calculate the cips
    logging.debug('What per batch like: %s', batch)
    a=[]
    b=[]
    og=[]
    d_og=[]
    feature_list=[]
    
    cip_index=[]
    all_ecom=[]
    for j,fe in enumerate(batch.num_edgecand):
      edge_id_from= sum(batch.num_edgecand[:j])
      edge_id_to = fe
      es=batch.edge_index.narrow(1,edge_id_from,edge_id_to)
      es=torch.transpose(es, 0, 1)
      eids=torch.arange(0,edge_id_to)
      etc=torch.combinations(eids)
      ecom=torch.cat((es[etc[:,0]],es[etc[:,1]]),dim=1)
      idx=(ecom[:,:2]==ecom[:,2:]).all(dim=1)
      
      cps,cpdst,ecom=shortest_dist_tensor(ecom,batch.gx,return_et=True,vdim=vdim)
      logging.debug('What the cps and cpdst: %s, %s',cps.shape,cpdst.shape)
      if (cps.view(-1)!=cps.view(-1)).any(): #remove rows that have NaN closest points #REY: NEED TO CHECK why this happened!!!!!

        logging.info('CPS got NaN !!!!!!!!!!!!!!!!!!!!')
        logging.info('CPS got NaN !!!!!!!!!!!!!!!!!!!!')
      if return_flist:
        #calculate some features for cps, 
        feas=calculate_features_from_cips(cps,cpdst)
        feature_list.append(feas.unsqueeze(0))
      #calculate normalization
      cm=cps.mean(dim=0)
      cstd=cps.std(dim=0)
      cdstd=cpdst.std(dim=0)
      logging.debug('What the cps mean and cpdst std: %s, %s',cm,cdstd)
      #logging.debug('What the cps: %s',cps)
      cstd[cstd==0]=1
      cps_orig=cps
      if list(cps.shape)[0]>1:
        cps=(cps-cm[None,:])/cstd[None,:]
      else:
        cps=cps-cm[None,:]
      
      if (cps!=cps).any():
        logging.info("ERROR: WARNING!! cps output NaN")
        logging.info("ERROR: WARNING!! cps output NaN")
        logging.info("cps NaN: %s",cps[(cps!=cps).any(dim=1)])
        logging.info("cps origin: %s",cps_orig[(cps!=cps).any(dim=1)])
        logging.info("cstd: %s",cstd)
      
      #logging.debug('What the cps-mean: %s',cps)            
      cpdst_org=cpdst
      if cdstd!=0 and cdstd==cdstd:
        cpdst=cpdst/cdstd
      og.append(cps_orig)
      d_og.append(cpdst_org)
      a.append(cps)
      b.append(cpdst)
      all_ecom.append(ecom)
      cip_index.extend([j for _ in range(list(cpdst.shape)[0]) ])
    
    #output cip result based on config parameter
    all_og=torch.cat(og,dim=0)
    all_ecom=torch.cat(all_ecom,dim=0)
    if input_features=="cip":
      all_cps=torch.cat(og,dim=0)
      all_cpdst=torch.cat(d_og,dim=0)
    elif input_features=="cip_distri":
      all_cps=torch.cat(a,dim=0)
      all_cpdst=torch.cat(b,dim=0)
    elif input_features=="cip+cip_distri":
      cp1=torch.cat(og,dim=0)
      cp2=torch.cat(a,dim=0)
      all_cps=torch.cat([cp1,cp2],dim=1)
      cd1=torch.cat(d_og,dim=0)
      cd2=torch.cat(b,dim=0)
      all_cpdst=torch.cat([ cd1[:,None],cd2[:,None] ],dim=1)
    else:
      raise Exception('config parameter "input_features" unknown' % self.input_features)
    
    cip_index=torch.tensor(cip_index)
    logging.debug('What are the all cps and all cpdst and cip_index: %s, %s, %s',all_cps.shape,all_cpdst.shape,cip_index.shape)
    ans=[all_cps,all_cpdst,cip_index]
    if return_vlist:
      ans.append(all_ecom)
    if return_flist:
      all_fea=torch.cat(feature_list,dim=0)
      ans.append(all_fea)
    return ans

def calculate_features_from_cips(cps,cpdst):
    # cps x,y,z mean, x,y,z variance, number of cps, the range of max x,y,z,d value - min x,y,z,d value
    ans=[]
    ans.append(cps.mean(dim=0))
    ans.append(cps.std(dim=0))
    ans.append(torch.tensor([cps.shape[0]],dtype=torch.float32))
    ans.append(cps.max(dim=0)[0]-cps.min(dim=0)[0])
    ans.append((cpdst.max()-cpdst.min()).unsqueeze(0))
    ans.append((cpdst.std()).unsqueeze(0))
    
    d=cps-cps.mean(dim=0)
    for i in range(d.shape[1]):
      x=torch.histc(d[:,i],bins=30)
      ans.append(x/sum(x))
    
    #print(f1,f2,f3,f4,f5,f6)
    feas=torch.cat(ans)
    #print(feas)
    return feas
      
    
    feas=torch.tensor([cps.mean(dim=0),cps.std(dim=0),cps.shape[0], 
      ])

def shortest_dist_tensor(et,vt,return_et=False,vdim=3,epsilon=10**-9): #et=edge index: [# edges,4], vt=hit_global_coordination: [# hits, 3]
    if (vt!=vt).all():
      raise Exception("Error: some coordinations in the input are NaN")
      #logging.info("Error: some coordinations in the input are NaN", vt[(vt!=vt).all(dim=1)])
      
    
    v1=vt[et[:,1],]-vt[et[:,0],]
    v2=vt[et[:,3],]-vt[et[:,2],]
    p1=vt[et[:,0],]
    p2=vt[et[:,2],]
    
    #remove the cases when v1,v2 are exactly the same
    leg_idx=(v1!=v2).any(dim=1)
    logging.debug("how v1 looks like: %s", v1.shape)
    v1=v1[leg_idx]
    v2=v2[leg_idx]
    p1=p1[leg_idx]
    p2=p2[leg_idx]
    et=et[leg_idx,:]
    logging.debug("how v1 prime looks like: %s", v1.shape)
    
    #logging.debug("how v1 looks like: %s", v1.shape)
    
    a=torch.bmm(v1.view(-1, 1, vdim), v1.view(-1, vdim, 1)).view(-1).double()
    b=torch.bmm(v1.view(-1, 1, vdim), v2.view(-1, vdim, 1)).view(-1).double()
    c=torch.bmm(v2.view(-1, 1, vdim), v2.view(-1, vdim, 1)).view(-1).double()
    w0=p1-p2
    d=torch.bmm(v1.view(-1, 1, vdim), w0.view(-1, vdim, 1)).view(-1).double()
    e=torch.bmm(v2.view(-1, 1, vdim), w0.view(-1, vdim, 1)).view(-1).double()

    if (a!=a).any():
      logging.debug("a is NaN: %s", a.shape)
      logging.info("a is NaN, with vector v1 as: %s", v1[a!=a])
      #logging.info("edge_list", et)
      #logging.info("hit_list", vt)
    if (b!=b).any():
      logging.info("b is NaN: %s", b.shape)
    if (c!=c).any():
      logging.info("c is NaN: %s", c.shape)
    if (d!=d).any():
      logging.info("d is NaN: %s", d.shape)
    if (e!=e).any():
      logging.info("e is NaN: %s", e.shape)

    s_c=(b*e-c*d)/((a*c-b*b)+epsilon)
    #s_c=(torch.dot(b,e)-torch.dot(c,d))/(torch.dot(a,c)-torch.dot(b,b))
    t_c=(a*e-b*d)/((a*c-b*b)+epsilon)  #element-wise multiplication

    #remove the cases when v1 and v2 are parallel 
    #s_c=s_c[a*c-b**2!=0]
    #t_c=t_c[a*c-b**2!=0]
    s_c=s_c[s_c==s_c]
    t_c=t_c[s_c==s_c]
    #t_c=(torch.dot(a,e)-torch.dot(b,d))/(torch.dot(a,c)-torch.dot(b,b))

    
    #logging.debug("how a looks like: %s", a.shape)
    #logging.debug("how b looks like: %s", b.shape)
    #logging.debug("how d looks like: %s", d.shape)
    #logging.debug("how s_c looks like: %s", s_c.shape)
    
    #logging.debug("s_c non-equal s_c2? : %s", (s_c!=s_c2).any())
    if (t_c!=t_c).any():
      er_id=(t_c!=t_c)
      logging.info("t_c is NaN, check a,b,c,d,e: %s,%s,%s,%s,%s", a[er_id],b[er_id],c[er_id],d[er_id],e[er_id])
      logging.info("t_c = ***/A-B, A: %s, B: %s",a[er_id]*c[er_id],b[er_id]*b[er_id])
    
    x1=p1+s_c.view(-1,1)*v1 #scalar op.
    x2=p2+t_c.view(-1,1)*v2 #scalar op.
    midd=x1+(x2-x1)/2
    v3p=x2-x1
    
    #calculate shortest distance by using the middle point (the closest point)
    #dstp=torch.sqrt(torch.bmm(v3p.view(-1, 1, vdim), v3p.view(-1, vdim, 1))) #there is some bugs with sqrt while doing gradient decent
    dstp=torch.bmm(v3p.view(-1, 1, vdim), v3p.view(-1, vdim, 1))
    dstp=dstp.view(-1)
    
    #calculate shortest distance by using the formula
    

    #logging.debug("midd is NaN: %s", (midd!=midd).any())
    midd=midd.float()
    dstp=dstp.float()
    #logging.debug("how v3p looks like: %s", v3p.shape)
    #logging.debug("how dstp looks like: %s", dstp.shape)
    if return_et:
      return [midd,dstp,et]
    return [midd,dstp]
    

def save_csv_from_list_of_tensors(l,csv_path,row_names=None): 
#tensors must be either 2d, 1d, or 1 value. The first element is the list must be the one with correct number of rows.
    a=None
    for i,x in enumerate(l):
      #print(i,x)
      if torch.is_tensor(x):
        x=x.cpu().detach().numpy()
      else:
        x=np.array(x)
      if i==0:
        a=x
      else:
        if len(x.shape)==1 or x.shape[0]==1:        #if the array is 1d or column based
          x=np.reshape(x,(-1,1))
        if x.shape[0]==1:          #if there is 1 value only
          x=np.transpose(np.array([[x[0][0]]*a.shape[0]]))
        if x.shape[0]!=a.shape[0]:
          raise Exception('the number of rows is not match, %s, %s:',a.shape,x.shape)
        a=np.concatenate((a,x),axis=1)
    
    #write into csv
    if not os.path.isfile(csv_path):
      with open(csv_path,'w+') as cf:
        if row_names!=None:
          cw=csv.writer(cf)
          cw.writerow(row_names)
    with open(csv_path,'a') as cf:
      cw=csv.writer(cf)
      cw.writerows(a)    


def shortest_dist(v1,v2,p1,p2): #v1, v2 is the 3D vector
  #method 1, calculate the closest points
  a=np.dot(v1,v1)
  b=np.dot(v1,v2)
  c=np.dot(v2,v2)
  w0=np.subtract(p1,p2)
  d=np.dot(v1,w0)
  e=np.dot(v2,w0)
  s_c=(b*e-c*d)/(a*c-b**2)
  t_c=(a*e-b*d)/(a*c-b**2)
  x1=np.add(p1,s_c*v1)
  x2=np.add(p2,t_c*v2)
  midd=np.add(x1, np.subtract(x2,x1)/2)
  v3p=np.subtract(x2,x1)
  v3p_l=np.linalg.norm(v3p)
  dstp=v3p_l
  
  #method part 2, calculate the shortest distance by cross product
  v3=np.cross(v1,v2) #the orthogonal vector
  v3_l=np.linalg.norm(v3) #the length of v3
  dst=np.dot(v3,np.subtract(p1,p2))/v3_l
  
  #Rey: There is an accuracy problem for two different methods (at least in 10^-6)
  #if not np.array_equal(v3p/v3p_l,v3/v3_l):  
  #  raise Exception('the function implementation is not correct')
  
  return [midd,dstp]


#=================== Rey: calculate ips from statistics method ==========================
def old_way_to_extract_cip_and_save_csv():
    if calculate_ips:
      self.logger.info('run and write into calculate_ips.csv')
      try: 
        self.model.edge_attr
      except:
        raise Exception('The edge attributions cannot be extracted')

      # Make predictions on this batch
      _ = self.model(batch)
      self.logger.debug('batch_edge_index shape: %s', batch.edge_index.shape)
      self.logger.debug('self.model.edge_attr shape: %s', self.model.edge_attr.shape)
      self.logger.debug('batch.x.shape: %s', batch.x.shape)
      self.logger.debug('# edges in batch: %s',batch.num_edgecand)
      self.logger.debug('max edge index in row: %s', max(batch.edge_index.cpu().numpy()[0]))
      
      # my func. calculate shortest distance and the closet point of two line 
      
      def shortest_dist(v1,v2,p1,p2): #v1, v2 is the 3D vector
        #method 1, calculate the closest points
        a=np.dot(v1,v1)
        b=np.dot(v1,v2)
        c=np.dot(v2,v2)
        w0=np.subtract(p1,p2)
        d=np.dot(v1,w0)
        e=np.dot(v2,w0)
        s_c=(b*e-c*d)/(a*c-b**2)
        t_c=(a*e-b*d)/(a*c-b**2)
        x1=np.add(p1,s_c*v1)
        x2=np.add(p2,t_c*v2)
        midd=np.add(x1, np.subtract(x2,x1)/2)
        v3p=np.subtract(x2,x1)
        v3p_l=np.linalg.norm(v3p)
        dstp=v3p_l
        
        #method part 2, calculate the shortest distance by cross product
        v3=np.cross(v1,v2) #the orthogonal vector
        v3_l=np.linalg.norm(v3) #the length of v3
        dst=np.dot(v3,np.subtract(p1,p2))/v3_l
        
        #Rey: There is an accuracy problem for two different methods (at least in 10^-6)
        #if not np.array_equal(v3p/v3p_l,v3/v3_l):  
        #  raise Exception('the function implementation is not correct')
        
        return [midd,dstp]
      
      
      
      
      #calculate mean and variance for each sample
      hs=batch.gx.cpu().numpy()
      eg_index=batch.edge_index.cpu().numpy()
      egs=self.model.edge_attr.detach().cpu().numpy()
      nhits=batch.num_hits.cpu().numpy()
      nedges=batch.num_edgecand.cpu().numpy()
      trigs=batch.trigger
      
      self.logger.info('Wrote batch: %s, with number of edge candidates: %s', str(i),nedges)
      
      for j,_ in enumerate(nhits):
        edge_id_from= sum(nedges[:j])
        edge_id_to = sum(nedges[:(j+1)])-1
        tg=trigs[j]
        sampleid=str(i)+"-"+str(j)
        IPC=namedtuple("IPC",['sample_id','trigger','gx_x',"gx_y","gx_z",'d','e1p','e2p']) #An tuple for an intersection point candidate
        sm=[]
        for e1 in range(edge_id_from,edge_id_to):
          for e2 in range(edge_id_from,edge_id_to):
            if e1 == e2:
              continue
            #first edge
            p1=hs[eg_index[0][e1]]
            p2=hs[eg_index[1][e1]]
            v1=np.subtract(p2,p1)
            #second edge
            p3=hs[eg_index[0][e2]]
            p4=hs[eg_index[1][e2]]
            v2=np.subtract(p4,p3)
            midd,d=shortest_dist(v1,v2,p1,p3)    
            sm.append(IPC(sampleid,tg,midd[0],midd[1],midd[2],d,egs[e1],egs[e2]))
        if sm==[]:
          print("something wrong here")
        #write into csv
        if not os.path.isfile(self.output_dir+"/calculate_ips.csv"):
          with open(self.output_dir+"/calculate_ips.csv",'w+') as cf:
            cw=csv.writer(cf)
            cw.writerow(list(IPC._fields))
        
        with open(self.output_dir+"/calculate_ips.csv",'a') as cf:
          cw=csv.writer(cf)
          cw.writerows(sm)
    #=================== Rey END: calculate ips from statistics method ==========================