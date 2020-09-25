import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from plotting import newfig, savefig

class neural_net(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100+1, n_output=1):
        super(neural_net, self).__init__()
        self.pathbatch = pathbatch
        self.fc_1 = nn.Linear(n_dim, 256)
        self.fc_2 = nn.Linear(256, 256)
        self.fc_3 = nn.Linear(256, 256)
        self.fc_4 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_output)

        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        with torch.no_grad():
            torch.nn.init.xavier_uniform(self.fc_1.weight)
            torch.nn.init.xavier_uniform(self.fc_2.weight)
            torch.nn.init.xavier_uniform(self.fc_3.weight)
            torch.nn.init.xavier_uniform(self.fc_4.weight)


    def forward(self, state, train=False):
        state = torch.sin(self.fc_1(state))   
        state = torch.sin(self.fc_2(state))  
        state = torch.sin(self.fc_3(state))  
        state = torch.sin(self.fc_4(state)) 
        fn_u = self.out(state)
        return fn_u

class FBSNN(nn.Module): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T, M, N, D, learning_rate):
        super().__init__()
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        self.fn_u = neural_net(pathbatch=M, n_dim=D+1, n_output=1)
        
        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)


    def phi_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return 0.05*(Y - torch.sum(X*Z, dim=1).unsqueeze(-1)) # M x 1
    
    def g_torch(self, X): # M x D
        return torch.sum(X**2, dim=1).unsqueeze(1) # M x 1

    def mu_torch(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return torch.zeros([self.M, self.D])  # M x D
        
    def sigma_torch(self, t, X, Y): # M x 1, M x D, M x 1
        return 0.4*torch.diag_embed(X) # M x D x D
        
    def net_u_Du(self, t, X): # M x 1, M x D
        inputs = torch.cat([t, X], dim=1)
        u = self.fn_u(inputs)
        Du = torch.autograd.grad(torch.sum(u), X, retain_graph=True)[0]  
        return u, Du

    def Dg_torch(self, X): # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0] # M x D

    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        Dt[:,1:,:] = dt
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return torch.from_numpy(t).float(), torch.from_numpy(W).float()

    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = torch.zeros(1)
        X_buffer = []
        Y_buffer = []
        
        t0 = t[:,0,:] # M x 1   
        W0 = W[:,0,:] # M x D
        X0 = torch.cat([Xi]*self.M) # M x D 
        X0.requires_grad = True
        Y0, Z0 = self.net_u_Du(t0,X0) # M x 1, M x D
        
        
        X_buffer.append(X0)
        Y_buffer.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            
            X1 = X0 + torch.matmul(self.mu_torch(t0,X0,Y0,Z0), t1-t0) + torch.matmul(self.sigma_torch(t0,X0,Y0), (W1-W0).unsqueeze(-1)).squeeze(2) # M x D 
            Y1_tilde = Y0 + self.phi_torch(t0,X0,Y0,Z0)*(t1-t0) + torch.sum(Z0*torch.matmul(self.sigma_torch(t0,X0,Y0), (W1-W0).unsqueeze(-1)).squeeze(2), dim=1).unsqueeze(1)
            Y1, Z1 = self.net_u_Du(t1,X1)
            
            loss = loss + torch.sum((Y1 - Y1_tilde)**2)
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
        
            X_buffer.append(X0)
            Y_buffer.append(Y0)
            
        loss = loss + torch.sum((Y1 - self.g_torch(X1))**2)
        loss = loss + torch.sum((Z1 - self.Dg_torch(X1))**2)

        X = torch.stack(X_buffer,dim=1) # M x N x D 
        Y = torch.stack(Y_buffer,dim=1) # M x N x 1
        
        return loss, X, Y, Y[0,0,0]

    def train(self, N_Iter=10):
        
        start_time = time.time()
        for it in range(N_Iter):
            
            t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' % 
                      (it, elapsed, loss, Y0_pred))
                start_time = time.time()

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        
        return X_star, Y_star

if __name__ == '__main__':
    
    M = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    learning_rate = 1e-3

    Xi = torch.from_numpy(np.array([1.0,0.5]*int(D/2))[None,:]).float()
    T = 1.0
    print(Xi.shape)

    model = FBSNN(Xi, T, M, N, D, learning_rate)

    model.train(N_Iter = 20000)


    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
        
    def u_exact(t, X): # (N+1) x 1, (N+1) x D
        r = 0.05
        sigma_max = 0.4
        return np.exp((r + sigma_max**2)*(T - t))*np.sum(X**2, 1, keepdims = True) # (N+1) x 1
    
    t_test = t_test.detach().numpy()
    X_pred = X_pred.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M,:,:],[-1,1]), np.reshape(X_pred[0:M,:,:],[-1,D])),[M,-1,1])
    print(Y_test[0,0,0])
    
    samples = 5
    
    plt.figure()
    plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1,:,0].T,Y_test[0:1,:,0].T,'r--',label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1,-1,0],Y_test[0:1,-1,0],'ko',label='$Y_T = u(T,X_T)$')
    
    plt.plot(t_test[1:samples,:,0].T,Y_pred[1:samples,:,0].T,'b')
    plt.plot(t_test[1:samples,:,0].T,Y_test[1:samples,:,0].T,'r--')
    plt.plot(t_test[1:samples,-1,0],Y_test[1:samples,-1,0],'ko')

    plt.plot([0],Y_test[0,0,0],'ks',label='$Y_0 = u(0,X_0)$')
    
    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    savefig('BSB.png', crop = False)
    
    
    errors = np.sqrt((Y_test-Y_pred)**2/Y_test**2)
    mean_errors = np.mean(errors,0)
    std_errors = np.std(errors,0)
    
    plt.figure()
    plt.plot(t_test[0,:,0],mean_errors,'b',label='mean')
    plt.plot(t_test[0,:,0],mean_errors+2*std_errors,'r--',label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('100-dimensional Black-Scholes-Barenblatt')
    plt.legend()
    
    savefig('BSB_error.png', crop = False)
        
