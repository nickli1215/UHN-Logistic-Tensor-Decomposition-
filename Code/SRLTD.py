import numpy as np
import pandas as pd
import os
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc
import argparse
import multiprocessing as mp
class SRLTD_bias:
    def __init__(self,num_factors=20,theta=0.25, lambda_d=0.625,lambda_t=0.625,lambda_sd=0.025, lambda_st=0.025, max_iter=100):
        self.num_factors = int(num_factors)
        self.theta = theta
        self.lambda_d = lambda_d
        self.lambda_sd = lambda_sd
        self.lambda_t = lambda_t
        self.lambda_st = lambda_st
        self.max_iter = max_iter
    def AGD_optimization (self, seed=None):
        if seed is None:
            self.U = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*np.random.normal(size=(self.num_targets, self.num_factors))
        else:
            prng = np.random.RandomState(seed)
            self.U = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_drugs, self.num_factors))
            self.V = np.sqrt(1/float(self.num_factors))*prng.normal(size=(self.num_targets, self.num_factors))
        
        self.B_u = np.ones((self.num_drugs))
        Vg_sum = np.zeros((self.num_targets,self.V.shape[1]))
        Ug_sum = np.zeros((self.num_drugs,self.U.shape[1]))
        
        Bu_sum = np.zeros((self.num_drugs))
        lastLog = self.BCE_loss()
        currDeltaLL = 1000
        for t in range(self.max_iter):
            V_grad= self.deriv(False)
            Vg_sum += np.square(V_grad)
            vec_step_size = self.theta / np.sqrt(Vg_sum)
            self.V -= vec_step_size*V_grad
                        
            U_grad, Bu_grad  = self.deriv(True)
            Ug_sum += np.square(U_grad)
            vec_step_size = self.theta/np.sqrt(Ug_sum)
            self.U -= vec_step_size * U_grad
            
            Bu_sum += np.square(Bu_grad)
            vec_step_size = self.theta/np.sqrt(Bu_sum)
            self.B_u -= vec_step_size*Bu_grad
            currentLog = self.BCE_loss()
            deltaLog = (currentLog-lastLog)/np.abs(lastLog)
            
            #if (np.abs(deltaLog)<1e-3):
                #break
            #if (t>50 and deltaLog>currDeltaLL):
               # break
            currDeltaLL = deltaLog  
            lastLog = currentLog
            print('Iteration ' + str(t+1) + ' Error: '+str(self.BCE_loss()))
    
    def BCE_loss(self):
        Q = np.empty((self.num_targets,self.num_drugs,self.num_drugs))
        #nxnxk tensor that represents the product of the three latent vectors
        for i in range(self.num_targets):
            for j in range(self.num_drugs):
                for k in range(self.num_drugs):
                    Q[i,j,k] = np.matmul(self.V[i]*self.U[j],self.U[k].T)
                    Q[i,j,k] += self.B_u[j] + self.B_u[k]
        P = self.sigmoid(Q)
        error = np.log(1+np.exp(P)) - self.Y*P
        error = error*self.mask
        return np.sum(error.flatten())/error.size
        
    
    def sigmoid(self, Q):
        return np.exp(Q)/(1+np.exp(Q))    
    
    def fix_model(self, mask, intTens, targetSim=None,drugSim=None,seed=None):
        self.num_targets,self.num_drugs, _ = intTens.shape
        self.mask = mask
        self.Y = intTens
        self.drugSim = drugSim
        self.targetSim = targetSim
        self.AGD_optimization()
    
    def deriv(self, drug):
        #If drug is true, return gradient for drug latent matrix
        #Otherwise return gradient for cell-line matrix
        Q = np.empty((self.num_targets,self.num_drugs,self.num_drugs))
        #nxnxk tensor that represents the product of the three latent vectors
        for i in range(self.num_targets):
            for j in range(self.num_drugs):
                for k in range(self.num_drugs):
                    Q[i,j,k] = np.matmul(self.V[i]*self.U[j],self.U[k].T)
                    Q[i,j,k] += self.B_u[j] + self.B_u[k]
        P = self.sigmoid(Q)
        A = (P - self.Y)*self.mask
        

        if (drug):
            #nxkxl tensor where J[i,k] is equal to the latent vector of drug i multiplied by latent vector of cell k       
            J = np.empty((self.num_targets,self.num_drugs,self.num_factors))
            for i in range(self.num_targets):
                for k in range(self.num_drugs):
                    J[i,k] = self.V[i]*self.U[k]
            U_grad = np.empty((self.num_drugs,self.num_factors))
            
            
            for i in range(self.num_drugs):
                U_grad[i] = np.sum(J*np.repeat(A[:,i,:][:,:,np.newaxis],self.num_factors,axis=2),axis=(0,1)) 
                U_grad[i] += np.sum(J[:,i]*np.repeat(A[:,i,i][:,np.newaxis],self.num_factors,axis=1),axis=0)
            if(type(self.drugSim)!=type(None)):
                S_d = self.drugSim - np.matmul(self.U,self.U.T)
                for i in range(self.num_drugs):
                    #U_grad[i] += self.lambda_sd*(2/np.linalg.norm(S_d,ord='fro'))*(np.sum(np.repeat(S_d[i][:,np.newaxis],self.num_factors,axis=1)*(-self.U),axis=0)+ S_d[i,i]*(-self.U[i]))
                    U_grad[i] += self.lambda_sd*(np.sum(np.repeat(S_d[i][:,np.newaxis],self.num_factors,axis=1)*(-self.U),axis=0)+ S_d[i,i]*(-self.U[i]))
            
            U_grad += 2*self.lambda_d*self.U
                
            
                
            return U_grad, np.sum(A,axis=(0,1))
        
        V_grad = np.empty((self.num_targets,self.num_factors))
        #nxnxl tensor where H[i,j] is equal to the latent vector of drug i multiplied by latent vector of drug j

        H = np.empty((self.num_drugs, self.num_drugs,self.num_factors))
        
        for i in range(self.num_drugs):
            for j in range(self.num_drugs):
                H[i,j] = self.U[i]*self.U[j]
        for k in range(self.num_targets):
            V_grad[k] = np.sum(H*np.repeat(A[k,:,:][:,:,np.newaxis],self.num_factors,axis=2),axis=(0,1))
            
        if (type(self.targetSim)!=type(None)):
            S_t = self.targetSim - np.matmul(self.V,self.V.T)
            for k in range(self.num_targets):
                #V_grad[k] += self.lambda_st*(2/np.linalg.norm(S_t,ord='fro'))*(np.sum(np.repeat(S_t[k][:,np.newaxis],self.num_factors,axis=1)*(-self.V),axis=0)+ S_t[k,k]*(-self.V[k]))
                V_grad[k] += self.lambda_st*(np.sum(np.repeat(S_t[k][:,np.newaxis],self.num_factors,axis=1)*(-self.V),axis=0)+ S_t[k,k]*(-self.V[k]))

        V_grad += 2*self.lambda_t*self.V
        return V_grad
    
    def predict(self):
        Q = np.empty((self.num_targets,self.num_drugs,self.num_drugs))
        #returns nxnxm tensor for predictions of scores
        for i in range(self.num_targets):
            for j in range(self.num_drugs):
                for k in range(self.num_drugs):
                    Q[i,j,k] = np.matmul(self.V[i]*self.U[j],self.U[k].T)
                    Q[i,j,k] += self.B_u[j] + self.B_u[k]
        return self.sigmoid(Q)
    def predict_new(self,new_V):
        num_targets = new_V.shape[0]
        Q = np.empty((num_targets,self.num_drugs,self.num_drugs))
        #returns nxnxm tensor for predictions of scores
        for i in range(num_targets):
            for j in range(self.num_drugs):
                for k in range(self.num_drugs):
                    Q[i,j,k] = np.matmul(new_V[i]*self.U[j],self.U[k].T)
                    Q[i,j,k] += self.B_u[j] + self.B_u[k]
        return self.sigmoid(Q)
    def get_U(self):
        return self.U
    
    def get_V(self):
        return self.V

def get_data(cells,drugs,dataset="MERCK_deepsynscores",threshold=30):
    data = []
    mask = []
    for cell in cells:
        cell_data = pd.read_csv("Data/"+dataset+"/"+cell+".csv",index_col=0).reindex(index=drugs,columns=drugs)
        for i in range(cell_data.shape[0]):
            cell_data.iloc[i,i] = False
        data.append(cell_data.to_numpy(dtype=np.float64))
    data = np.array(data)
    mask= ~np.isnan(data)
    data = data>threshold
    return mask, data

def get_new_V(gene_exprs,train_cells,test_cells,model):
    scaler = StandardScaler()
    scaler.fit(gene_exprs.loc[train_cells])
    X_train = scaler.transform(gene_exprs.loc[train_cells])
    X_test =  scaler.transform(gene_exprs.loc[test_cells])
    regr = ElasticNet(random_state=0)
    regr.fit(X_train,model.get_V())
    new_V = regr.predict(X_test)
    return new_V

def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('f',type=int,help="fold_num")
    parser.add_argument('i',type=int,help="fold_num")
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    outer_fold = args.f
    inner_fold=args.i
    labels = pd.read_csv("Data/labels_w_cells.csv",index_col=0)
    cells = np.unique(labels['cell_line'])
    drugs = np.unique(labels[['drug_a_name','drug_b_name']].to_numpy().flatten())
    train_cells=pd.read_csv("Data/Cell_Splits/train_"+str(outer_fold)+"_"+str(inner_fold)+".csv",header=None).to_numpy().T[0]
    val_cells=pd.read_csv("Data/Cell_Splits/val_"+str(outer_fold)+"_"+str(inner_fold)+".csv",header=None).to_numpy().T[0]
    train_mask,train_data=get_data(train_cells,drugs)
    val_mask,val_data=get_data(val_cells,drugs)
    cell_sim=pd.read_csv("Data/DeepSyn_sim.csv",index_col=0).loc[train_cells,train_cells].reindex(index=train_cells,columns=train_cells)
    drug_sim=pd.read_csv("Data/drugfp_sim.csv",index_col=0).reindex(index=drugs,columns=drugs)
    hyperparameters=pd.read_csv("Data/MERCK_results/SRLTD_"+str(outer_fold)+"_"+str(inner_fold)+".csv",index_col=0)
    gene_expression=pd.read_csv("Data/deepsynergy_gene_expression.csv",index_col=0)
    data=[]
    for i,j in hyperparameters.loc[hyperparameters['AUPR'].isnull()].iterrows():
        num_factors = int(j[0])
        theta= j[1]
        lambda_d=j[2]
        lambda_t=j[3]
        lambda_st=j[4]
        lambda_sd=j[5]
        model=SRLTD_bias(num_factors=num_factors,theta=theta,lambda_d=lambda_d,lambda_t=lambda_t,lambda_sd=lambda_sd,lambda_st=lambda_st,max_iter=100)
        model.fix_model(train_mask,train_data,targetSim=cell_sim.to_numpy(),drugSim=drug_sim.to_numpy())
        new_V = get_new_V(gene_expression,train_cells,val_cells,model)
        y_val = val_data.flatten()[np.where(val_mask.flatten())]
        y_pred = model.predict_new(new_V).flatten()[np.where(val_mask.flatten())]
        precision,recall,thresholds=precision_recall_curve(y_val,y_pred)
        hyperparameters.loc[i,'AUPR']=auc(recall,precision)
        hyperparameters.to_csv("Data/MERCK_results/SRLTD_"+str(outer_fold)+"_"+str(inner_fold)+".csv")