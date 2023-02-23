import os, time, importlib, errno
import matplotlib.pyplot as plt 

import numpy as np

# from dt_utils import Timer
#from networks import HebbDiags

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset

#%%#################
### Base Classes ###
####################
   
class NetworkBase(nn.Module): 
    def __init__(self):
        """Subclasses must either implement self.loss_fn or override self.average_loss()"""
        super(NetworkBase, self).__init__()
        self.eval() #start the module in evaluation mode
        self.hist = None
        self.name = self.__class__.__name__
        self.loss_fn = None 
        self.acc_fn = None
        self.learning_rate = None
        self.verbose = True
    
    def fit(self, train_fn, *args, **kwargs):
        if train_fn in ('dataset', 'sequence',):
            train_fn = train_dataset            
        elif train_fn == 'infinite':
            train_fn = train_infinite           
        elif train_fn == 'curriculum':
            train_fn = train_curriculum
        elif train_fn == 'multiR':
            train_fn = train_multiR_curriculum
        elif hasattr(train_fn, '__call__'):
            pass        
        else:
            ValueError("train_fn must be a function or valid keyword")
        
        # If anything about the training has changed, this should be triggered to mark a transition.
        # Mostly this is important for computing early stop conditions, to make sure the network is not
        # comparing loss on the new set to loss computed from the old set.
        self.newThresh = kwargs.pop('newThresh', True)

        self.learning_rate = kwargs.pop('learningRate', 1e-3)
        init_string = 'Train parameters:'
        init_string += '\n  Loss: XE // LR: {:.2e} '.format(self.learning_rate)

        self.weightReg = kwargs.pop('weightReg', None)
        self.regLambda = kwargs.pop('regLambda', None)
        if self.weightReg is not None:
            init_string += '// Weight reg: {}, coef: {:.1e} '.format(self.weightReg, self.regLambda)      
        else:
            init_string += '// Weight reg: None '

        self.gradientClip = kwargs.pop('gradientClip', None)
        if self.gradientClip is not None:
            init_string += '// Gradient clip: {:.1e}'.format(self.gradientClip)      
        else:
            init_string += '// Gradient clip: None'

        self.paramSimilarity = kwargs.pop('paramSimilarity', None)
        self.paramSimLambda = kwargs.pop('paramSimLambda', None)
        if self.paramSimilarity is not None:
            init_string += '// Param sim: {}, coef: {:.1e} '.format(self.paramSimilarity, self.paramSimLambda)      

        folder = os.path.normpath(kwargs.pop('folder', ''))
        filename = kwargs.pop('filename', None)
        if filename is not None and not os.path.commonprefix([os.getcwd()[::-1],folder[::-1]]):
            if folder != '' and not os.path.exists(folder):
                os.makedirs(folder)
            os.chdir(folder)
        self.autosave = Autosave(self, filename)
        if kwargs.pop('overwrite', False):
            self.autosave.lastSave = 0
        self.autosave(force=True)
        if filename is not None and folder is not None:
#            plt.ioff() #if using SummaryWriter, don't display plots to desktop
#            plt.switch_backend('agg')
            writerPath = os.path.join('tensorboard', self.autosave.filename[:-4])
            if not os.path.commonprefix([os.getcwd()[::-1],folder[::-1]]):
                writerPath = os.path.join(folder, writerPath)
            self.writer = SummaryWriter(writerPath) #remove extension from filename

        if self.verbose and self.newThresh: # Only prints training details if newThreshold
            print(init_string)

        self.mointorFreq = kwargs.pop('monitorFreq', 10)        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)       
        self.train() #put Module in train mode (e.g. for dropout)
        
        scheduler = kwargs.pop('scheduler', '')
        self.scheduler = None
        if scheduler == 'reducePlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        
        # with Timer() as timer:
        early_stop = train_fn(self, *args, **kwargs)                   
        self.eval() #put Module back in evaluation mode
        
        self.autosave(force=True)
        # self.hist['time'] = timer.elapsed 

        return early_stop       
    
    def _train_epoch(self, trainData, validBatch=None, batchSize=1, earlyStop=True, earlyStopValid=False, validStopThres=None, 
                     trainOutputMask=None, validOutputMask=None, minMaxIter=(-2, 1e7)):
        # print('New epoch...')
        for b in range(0, trainData.tensors[0].shape[0], batchSize):  #trainData.tensors[0] is shape [B,T,Nx]
            trainBatch = trainData[b:b+batchSize,:,:] 
            
            if trainOutputMask is not None: 
                trainOutputMaskBatch = trainOutputMask[b:b+batchSize,:,:] 
            else:
                trainOutputMaskBatch = None

            self.optimizer.zero_grad()

            out = self.evaluate(trainBatch) #expects shape [B,T,Nx], out: [B,T,Ny]
            loss = self.average_loss(trainBatch, out=out, outputMask=trainOutputMaskBatch)
            loss.backward()
            
            # Zero NaNs in gradients. This can sometimes pop up with BNNs.
            for name, param in self.named_parameters():
                param.grad = torch.nan_to_num(param.grad, 0.0)

            if self.gradientClip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradientClip)

            self.optimizer.step() 
            
            # Note: even though this is called every batch, only runs validation batch every monitor_freq batches
            self._monitor(trainBatch, validBatch=validBatch, out=out, loss=loss, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask) 
            # self._monitor(trainBatch, validBatch=validBatch, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask)  
 
            # if earlyStopValid and len(self.hist['valid_loss'])>1 and self.hist['valid_loss'][-1] > self.hist['valid_loss'][-2]:
            STEPS_BACK = 10
            # Stop if the avg_valid_loss has asymptoted (or starts to increase)
            # (since rolling average is 10 monitors, 2*STEPS_BACK makes sure there are two full averages available for comparison,
            #  subtracting self.hist['monitor_thresh'][-1] prevents this threshold from being tested when a network continues training,)
            if self.hist['iter'] > minMaxIter[0]: # Only early stops when above minimum iteration count
                if earlyStopValid and (len(self.hist['iters_monitor']) - self.hist['monitor_thresh'][-1]) > 2*STEPS_BACK:
                    if 1.0 * self.hist['avg_valid_loss'][-STEPS_BACK] < self.hist['avg_valid_loss'][-1]:
                        print('  Early Stop: avg_valid_loss saturated, current (-1): {:.2e}, prev (-{}): {:.2e}, acc: {:.2f}'.format(
                            self.hist['avg_valid_loss'][-1], STEPS_BACK, self.hist['avg_valid_loss'][-STEPS_BACK], self.hist['avg_valid_acc'][-1]))
                        return True
                if validStopThres is not None and self.hist['avg_valid_acc'][-1] > validStopThres:
                    print('  Early Stop: valid accuracy threshold reached: {:.2f}'.format(
                        self.hist['avg_valid_acc'][-1]
                    ))
                    return True
                if self.hist['iter'] > minMaxIter[1]: # Early stop if above maximum numbers of iters
                    print('  Early Stop: maximum iterations reached, acc: {:.2f}'.format(
                        self.hist['avg_valid_acc'][-1]
                    ))
                    return True
            # if earlyStop and sum(self.hist['train_acc'][-5:]) >= 4.99: #not a proper early-stop criterion but useful for infinite data regime
            #     return True
        return False  

    def evaluate(self, batch): #TODO: figure out nice way to infer shape of y and don't pass in
        """batch is (X,Y) tuple of Tensors, with X.shape=[T,B,N] or [T,N], Y.shape=[T,1]"""
        #TODO: automatically call batched (if it exists) or per-sample version of forward()
        out = torch.empty_like(batch[1]) #shape=[T,B,Ny] if minibatched, otherwise [T,Ny] 
        for t,x in enumerate(batch[0]): #shape=[T,B,Nx] or [T,Nx]
            out[t] = self(x) #shape=[B,Nx] or [Nx]
        return out


    def evaluate_debug(self, batch):
        """Override this"""
        raise NotImplementedError


    def accuracy(self, batch, out=None, outputMask=None):
        """batch is (x,y) tuple of Tensors, with x.shape=[T,B,Nx] or [T,Nx]"""
        if out is None:
            out = self.evaluate(batch)
        if outputMask is None:
            return self.acc_fn(out, batch[1])
        else: # Modify output by the mask
            masked_y = batch[1][outputMask[:, :, 0]].squeeze(-1) # [B*T_mask, 1] -> [B*T_mask]
            masked_out = out[outputMask[:, :, 0]] # [B*T_mask, Ny]

            return self.acc_fn(masked_out, masked_y)


    def average_loss(self, batch, out=None, outputMask=None):
        """batch is (x,y) tuple of Tensors, with x.shape=[T,B,Nx] or [T,Nx]"""
        B = batch[0].shape[0]
        if out is None:
            out = self.evaluate(batch)
        
        # Computes regularization term
        if self.weightReg in ('L1', 'L2'):
            # Omits certain paramters from regularization (e.g. sigma and eta in HebbNets)
            p_reg = []
            for p in self.parameters():
                if torch.prod(torch.tensor(p.shape)) > 1:
                    p_reg.append(p)
    
            if self.weightReg == 'L2':
                l2_norm = sum(p.pow(2.0).sum()
                              for p in p_reg)
                reg_term = self.regLambda * l2_norm/2
            elif self.weightReg == 'L1':
                l1_norm = sum(p.abs().sum()
                              for p in p_reg)
                reg_term = self.regLambda * l1_norm
                # print('Batch size: {}, L1 Norm {:0.5f}, Reg Term {:0.5f}'.format(B, l1_norm, reg_term))
        else:
            reg_term = 0.0

        if self.paramSimilarity is not None:
            param_vecs = []
            # Only takes certain named parameters, for now this is simply hardcoded to what we need, but would be nice to adapt later
            for name, param in self.named_parameters():
                if param.requires_grad and name[-4:] in ('_eta', '_lam'):    
                    param_vecs.append(param) # Each param should be (1, 1, N)

            # Each neuron gets a 2d vector, (eta, lam)
            param_vecs = torch.cat(param_vecs, dim=1) # (1, 2, N)
            param_vecs = torch.transpose(param_vecs, 1, 2) # (1, N, 2)
            # Computes pairewise distance matrix, then takes only upper diag, then sums all quantities
            dists = torch.sum(torch.triu(torch.cdist(param_vecs, param_vecs, p=2).squeeze(0), diagonal=1)) # (1, N, N) before squeeze

            param_sim_term = self.paramSimLambda * dists
        else:
            param_sim_term = 0.0
        
        if outputMask is None:
            print('XE probably wont work here')
            return self.loss_fn(out, batch[1]) + reg_term
        else: # Modify output by the mask
            # Last index of outputMask not needed because labels are [B, T]
            masked_y = batch[1][outputMask[:, :, 0]].squeeze(-1) # [B*T_mask, 1] -> [B*T_mask]
            masked_out = out[outputMask[:, :, 0]] # [B*T_mask, Ny]

            # print('outputMask shape:', outputMask.shape)
            # print('out shape:', out.shape)
            # print('y shape:', batch[1].shape)
            # print('masked out shape:', masked_out.shape)
            # print('masked y shape:', masked_y.shape)

            # print('masked out:', masked_out)
            # print('masked y:', masked_y)

            # print('type masked out:', masked_out.type())
            # print('type masked y:', masked_y.type())

            # Flatten over batch and temporal indices
            return self.loss_fn(masked_out, masked_y) + reg_term + param_sim_term

    
    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None, trainOutputMaskBatch=None, validOutputMask=None):
        if self.hist is None: # Initialize self.hist if needed
            self.hist = {}
        if 'epoch' not in self.hist: # This allows self.hist to be initialized in children classes first, and then all default keys are added here  
            self.hist['epoch'] = 0
            self.hist['iter'] = -1 # gets incremented when _monitor() is called
            self.hist['iters_monitor'] =  []
            self.hist['monitor_thresh'] =  [0,] # used to calculate rolling average loss/accuracy
            self.hist['train_loss'] =  [] 
            self.hist['train_acc'] =  []
            self.hist['grad_norm'] =  []

            if validBatch:
                self.hist['valid_loss'] = []
                self.hist['valid_acc']  = []
                self.hist['avg_valid_loss'] = []
                self.hist['avg_valid_acc']  = []
                
            self._monitor(trainBatch, validBatch=validBatch, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask)
        else: 
            if self.verbose:
                print('Network already partially trained. Last iter {}'.format(self.hist['iter']))    

            # Will be used in _monitor to calculate the rolling loss relative to new iter value
            self.hist['monitor_thresh'].append(len(self.hist['iters_monitor']))
            # Does an initial monitor to prevent immediate stopping based off of parameters
            self._monitor(trainBatch, validBatch=validBatch, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask, runValid=True)


    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None, trainOutputMaskBatch=None, validOutputMask=None, runValid=False):  
        #TODO: rewrite with same format as SummaryWriter and automatically write to both 
        self.hist['iter'] += 1 
        
        if self.hist['iter']%self.mointorFreq == 0 or runValid: #TODO: allow choosing monitoring interval
            if out is None:
                out = self.evaluate(trainBatch)                
            if loss is None:
                loss = self.average_loss(trainBatch, out, outputMask=trainOutputMaskBatch)
            if acc is None:
                acc = self.accuracy(trainBatch, out, outputMask=trainOutputMaskBatch)
            gradNorm = self.grad_norm()
            
            self.hist['iters_monitor'].append(self.hist['iter'])
            self.hist['train_loss'].append(loss.item())
            self.hist['train_acc'].append(acc.item()) 
            self.hist['grad_norm'].append(gradNorm)            
            if hasattr(self, 'writer'):
                self.writer.add_scalar('train/loss', loss.item(), global_step=self.hist['iter'])   
                self.writer.add_scalar('train/acc', acc.item(), global_step=self.hist['iter'])   
                self.writer.add_scalar('info/grad_norm', gradNorm, global_step=self.hist['iter'])
            displayStr = 'Iter:{} lr:{:.3e} grad:{:.3f} train_loss:{:.4f} train_acc:{:.3f}'.format(
                self.hist['iter'], self.optimizer.param_groups[0]['lr'], gradNorm, loss, acc)                    
            
            if validBatch is not None:
                valid_out = self.evaluate(validBatch)
                valid_loss = self.average_loss(validBatch, out=valid_out, outputMask=validOutputMask)  
                valid_acc = self.accuracy(validBatch, out=valid_out, outputMask=validOutputMask) 
                
                if self.scheduler is not None:
                    self.scheduler.step(valid_loss) # Scheduler will decrease LR if we hit a plateau
                    
                self.hist['valid_loss'].append(valid_loss.item())                
                self.hist['valid_acc'].append(valid_acc.item())

                # Rolling average values, with adjustable window to account for early training times
                N_AVG_WINDOW = 10
                avg_window = min((N_AVG_WINDOW, len(self.hist['iters_monitor'])-self.hist['monitor_thresh'][-1]))
                # print('Winodw: {} - Loss, acc lengths {}, {}'.format(avg_window,
                #     len(self.hist['valid_loss']),
                #     len(self.hist['valid_acc'])))
                self.hist['avg_valid_loss'].append(np.mean(self.hist['valid_loss'][-avg_window:]))
                self.hist['avg_valid_acc'].append(np.mean(self.hist['valid_acc'][-avg_window:]))

                if hasattr(self, 'writer'):                                 
                    self.writer.add_scalar('validation/acc', acc.item(), global_step=self.hist['iter']) 
                    self.writer.add_scalar('validation/acc', acc.item(), global_step=self.hist['iter']) 
                displayStr += ' valid_loss:{:.4f} valid_acc:{:.3f}'.format(valid_loss, valid_acc)

            if self.verbose:
                print(displayStr)
            self.autosave()            
            return out 
           
            
    def grad_norm(self):
        return torch.cat([p.reshape(-1,1) for p in self.parameters() if p.requires_grad]).norm().item()
        
           
    def save(self, filename, overwrite=False):
        """Doing this the recommended way, see: 
            https://pytorch.org/docs/stable/notes/serialization.html#recommend-saving-models
        To load, initialize a new network of the same type and do net.load(filename) or use load_from_file()        
        """   
        device = next(self.parameters()).device
        if device != torch.device('cpu'):   
            self.to('cpu') #make sure net is on CPU to prevent loading issues from GPU 
        
        # directory = os.path.split(filename)[0]
        # if directory != '' and not os.path.exists(directory):
        #     os.makedirs(directory)
            
        # if not overwrite:
        #     base, ext = os.path.splitext(filename)
        #     n = 2
        #     while os.path.exists(filename):
        #         filename = '{}_({}){}'.format(base, n, ext)
        #         n+=1
        
        state = self.state_dict()
        state.update({'hist':self.hist})
        if self.verbose:
            print('  Saving net to: {}'.format(filename))
        torch.save(state, filename) 
        
        if device != torch.device('cpu'):
            self.to(device) #put the net back to device it started from
        return filename

    
    def load(self, filename):
        state = torch.load(filename)
        self.hist = state.pop('hist')
        
        try:
            self.load_state_dict(state, strict=False)
        except:
            for k in state.keys():
                checkptParam = state[k]
                                               
                kList = k.split('.')
                localParam = getattr(self, kList[0])
                for k in kList[1:]:
                    localParam = getattr(localParam, k)    
                    
                if localParam.shape != checkptParam.shape:
                    print('WARNING: shape mismatch between local {} ({}) and checkpoint ({}). Setting local {} to be {}'.format(k, localParam.shape, checkptParam.shape, k, checkptParam.shape))
                    localParam.data = torch.empty_like(checkptParam)
            self.load_state_dict(state, strict=False)



class StatefulBase(NetworkBase):
    """Networks that have an internal state that evolves with time, e.g. RNNs, LSTMs, HebbNets"""
#    def __init__(self):
#        """reset_state() needs to be called *after* subclass init is done. This is possible, 
#        but makes code harder to read. 
#
#        To do so, create an abstract __init_params__ method in NetworkBase, then run 
#        __init_params__(*params) inside of NetworkBase.__init__(*params), which is called by
#        super() and thus executes before self.reset_state(). Like this, the __init_params__  
#        method will be overridden in every subclass, but it would be called by the abstract 
#        class under the hood, making it somewhat obscure. 
#
#        Instead, just remember to do reset_state() in the subclass."""
#        super(StatefulNetworkBase,self).__init__(*params)
#        self.reset_state() 


    def reset_state(self):
        raise NotImplementedError


    def evaluate(self, batch, preserveState=False):
        """NOTE: current state of A will be lost!"""        
#        TODO: save state before running evaluation, then run, then restore? 
#        e.g.
#        if preserveState: #how to make this save all state vars w/o having to name them? --> use register_buffer()!
#           OLDSTATEVARS = self.STATEVARS.clone().detach() 
#        <eval code> #this will change all the state variables
#        if preserveState:
#           self.STATEVARS = OLDSTATEVARS
#        
#        or 
#        with StatePreserver(preserveState, self.STATEVARS.clone.detach()):
#            <eval code>                
        self.reset_state()
        out = super(StatefulBase,self).evaluate(batch)
        return out


#    def evaluate_batched(self, data, preserveState=False):
#        """NOTE: current state of A will be lost!"""        
##        TODO: save state before running evaluation, then run, then restore? 
##        e.g.
##        if preserveState: #how to make this save all state vars w/o having to name them?..
##           OLDSTATEVARS = self.STATEVARS.clone().detach() 
##        <eval code> #this will change all the state variables
##        if preserveState:
##           self.STATEVARS = OLDSTATEVARS
##        
##        or 
##        with StatePreserver(preserveState, self.STATEVARS.clone.detach()):
##            <eval code>                
#        batchsize = data[0].shape[1]
#        self.reset_state(batchsize)
#        out = super(StatefulBase,self).evaluate(data)
#        return out  


#%%#############
### Training ###
################
    
def train_dataset(net, trainData, validBatch=None, epochs=100, batchSize=1, validStopThres=None, earlyStopValid=False, 
    trainOutputMask=None, validOutputMask=None, minMaxIter=(-2, 1e7)): 
    """trainData is a TensorDataset"""    
    early_stop = False

    # Only need to _monitor_init if a new threshold has begun
    if net.newThresh:
        # _monitor_init called with just the zeroth batch index 
        trainOutputMaskBatch = trainOutputMask[0:batchSize, :, :] if trainOutputMask is not None else None 
        net._monitor_init(trainData[0:batchSize,:,:], validBatch=validBatch, trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask)
    while net.hist['epoch'] < epochs:
        net.hist['epoch'] += 1
        converged = net._train_epoch(trainData, validBatch=validBatch, batchSize=batchSize, 
                                     validStopThres=validStopThres, earlyStopValid=earlyStopValid, 
                                     trainOutputMask=trainOutputMask, validOutputMask=validOutputMask,
                                     minMaxIter=minMaxIter)  
        if converged:
            # print('Converged, stopping early.')
            early_stop = True
            break                                      

        trainData, trainOutputMask = shuffle_dataset(trainData, trainOutputMask)

    return early_stop

def shuffle_dataset(dataSet, dataSetMask=None):
    """ Shuffles a dataset over its batch index """
    
    # print('Shuffling data...')
    x, y = dataSet.tensors
    assert x.shape[0] == y.shape[0] # Checks batch indexes are equal
    shuffle_idxs = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idxs)
    x = x[shuffle_idxs, :, :]
    y = y[shuffle_idxs, :, :]

    newDataSet = TensorDataset(x, y)
    
    if dataSetMask is not None:
        assert x.shape[0] == dataSetMask.shape[0]
        dataSetMask = dataSetMask[shuffle_idxs, :, :]
    
    return newDataSet, dataSetMask


def train_infinite(net, gen_data, iters=float('inf'), batchSize=None, earlyStop=True):     
    trainBatch = gen_data()[:,0,:] if batchSize is None else gen_data()[:,:batchSize,:] 
    net._monitor_init(trainBatch)
    
    iter0 = net.hist['iter'] 
    
    while net.hist['iter'] < iters:
        trainCache = gen_data() #generate a large cache of data, then minibatch over it
        converged = net._train_epoch(trainCache, batchSize=batchSize, earlyStop=earlyStop)    
        
        #TODO: this is a hack to force the net to add at least 5 entries to hist['acc'] 
        #before evaluating convergence. Ensures that if we're starting from a converged network
        #we re-evaluate on 5 *new* entries.        
        if net.hist['iter'] < iter0+50: #hist['acc'] updated every 10 iters      
            converged = False
        
        if converged:
            print('Converged, stopping early.')
            break   

      
def train_curriculum(net, gen_data, iters=float('inf'), itersToQuit=2e6, batchSize=None, R0=1, Rf=float('inf'), increment=lambda R:R+1):          
    R = R0
    trainBatch = gen_data(R)[:,0,:] if batchSize is None else gen_data(R)[:,:batchSize,:] 
    net._monitor_init(trainBatch)
    if 'increment_R' not in net.hist:
        net.hist['increment_R'] = [(0, R)]
    itersSinceIncrement = 0
    latestIncrementIter = net.hist['iter']

    converged = False
    while net.hist['iter'] < iters and R < Rf and itersSinceIncrement < itersToQuit:
        if hasattr(net, 'writer'):
            net.writer.add_scalar('info/R', R, net.hist['iter'])
        if converged:
            R = increment(R)
            latestIncrementIter = net.hist['iter']
            print('Converged. Setting R<--{} \n'.format(R))
            net.hist['increment_R'].append( (net.hist['iter'], R) )    
            net.autosave(force=True)                      
                 
        trainCache = gen_data(R)   
        converged = net._train_epoch(trainCache, batchSize=batchSize)  
        itersSinceIncrement = net.hist['iter'] - latestIncrementIter
        
        #TODO: this is a hack to force the net to add at least 5 entries to hist['acc'] 
        #before evaluating convergence. Ensures that once it's converged on R, it doesn't
        #assume it's converged on R+1 since the convergence depends on the last 5 entries of hist['acc']        
        if net.hist['iter'] < net.hist['increment_R'][-1][0]+50:
            converged = False 


def train_multiR_curriculum(net, gen_data, Rlo=1, Rhi=2, spacing=range, batchSize=1, itersToQuit=2e6, increment=None ):   
    """
    Default: train on [1..Rchance]
    """
    if increment is None:
        def increment(Rlo, Rhi):
            return Rlo, Rhi+1           
    
    Rlist = spacing(Rlo, Rhi)
    trainBatch = gen_data(Rlist)[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
    validBatch = gen_data([Rlist[-1]])[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
    net._monitor_init(trainBatch, validBatch)
    if 'increment_R' not in net.hist:
        net.hist['increment_R'] = [(0, Rlist[-1])]
    itersSinceIncrement = 0
    latestIncrementIter = net.hist['iter']
        
    converged = False
    itersSinceIncrement = 0
    while itersSinceIncrement < itersToQuit:
        if hasattr(net, 'writer'):
            net.writer.add_scalar('info/Rlo', Rlist[0], net.hist['iter'])
            net.writer.add_scalar('info/Rhi', Rlist[-1], net.hist['iter'])
        if converged:
            Rlo, Rhi = increment(Rlo, Rhi)
            Rlist = spacing(Rlo, Rhi)
            latestIncrementIter = net.hist['iter']
            print('acc(Rchance)>0.55. Setting Rlist=[{}...{}] \n'.format(Rlist[0], Rlist[-1]))
            net.hist['increment_R'].append( (net.hist['iter'], Rlist[0], Rlist[-1]) )    
            net.autosave(force=True)  
        itersSinceIncrement = net.hist['iter'] - latestIncrementIter    
   
        trainCache = gen_data(Rlist)   
        
        #TODO: this is dumb, I'm generating 1000x more data than I'm using for validation
        validBatch = gen_data([Rlist[-1]])[:,0,:] if batchSize is None else gen_data(Rlist)[:,:batchSize,:] 
        converged = net._train_epoch(trainCache, validBatch, batchSize=batchSize, earlyStop=False, validStopThres=0.55)
               

#%%############
### Metrics ###
###############
        
def nan_mse_loss(out, y, reduction='mean'):
    """Computes MSE loss, ignoring any nan's in the data"""
    idx = ~torch.isnan(y)
    return F.mse_loss(out[idx], y[idx], reduction=reduction)


def nan_bce_loss(out, y, reduction='mean'):
    idx = ~torch.isnan(y)
    return F.binary_cross_entropy(out[idx], y[idx], reduction=reduction)


def nan_recall_accuracy(out, y):
    """Computes accuracy on the recall task, ignoring any nan's in the data"""
    idx = ~torch.isnan(y).all(dim=-1)
    return (out[idx].sign()==y[idx]).all(dim=-1).float().mean()    
 
    
def binary_thres_classifier_accuracy(out, y, thres=0.5):
    """Accuracy for binary-classifier-like task"""
    return ((out>thres) == y.bool()).float().mean()


def binary_classifier_accuracy(out, y):
    """Accuracy for binary-classifier-like task"""
    return (out.round() == y.round()).float().mean() #round y in case using soft labels

def xe_classifier_accuracy(out, y):
    """Accuracy for binary-classifier-like task"""
    # print('out shape', out.shape)
    # print('y shape', y.shape)

    return (torch.argmax(out, dim=-1) == y).float().mean() 

def nan_binary_classifier_accuracy(out, y):
    idx = ~torch.isnan(y)
    return binary_classifier_accuracy(out[idx], y[idx])    

    
#%%############    
### Helpers ### 
############### 
import numpy as np #TODO: move to torch, remove numpy  
    
def check_dims(W, B=None):
    """Verify that the dimensions of the weight matrices are compatible"""
    dims = [W[0].shape[1]]
    for l in range(len(W)-1):
        assert(W[l].shape[0] == W[l+1].shape[1]) 
        dims.append( W[l].shape[0] )
        if B: 
            assert(W[l].shape[0] == B[l].shape[0])
    if B: 
        assert(W[-1].shape[0] == B[-1].shape[0]) 
    dims.append(W[-1].shape[0])
    return dims
        

def random_weight_init(dims, bias=False):
    W,B = [], []
    for l in range(len(dims)-1):
        # W.append(np.random.randn(dims[l+1], dims[l])/np.sqrt(dims[l]))
        xavier_val = np.sqrt(6)/np.sqrt(dims[l] + dims[l+1])
        W.append(np.random.uniform(low=-xavier_val, high=xavier_val, size=(dims[l+1], dims[l])))
        if bias:        
            # B.append( np.random.randn(dims[l+1]) )
            xavier_val_b = np.sqrt(6)/np.sqrt(dims[l+1])
            B.append(np.random.uniform(low=-xavier_val_b, high=xavier_val_b, size=(dims[l+1])))
        else:
            B.append( np.zeros(dims[l+1]) )
    check_dims(W,B) #sanity check
    return W,B
 
     
class Autosave():
    def __init__(self, net, filename, saveInterval=3600):
        self.filename = filename
        self.lastSave = -1
        self.net = net
        self.saveInterval = saveInterval
        self.it = 0
             
    def __call__(self, force=True):
        if self.filename is not None and (force or time.time()-self.lastSave>self.saveInterval):
            filename = self.filename + '_' + str(self.it) + '.pt'
            self.net.save(filename)
            self.lastSave = time.time()
            self.it += 1


def load_from_file(fname, NetClass=None, dims=None):
    """If NetClass and dims not provided, filename format must be of the format NetName[dims]*.pkl 
    where NetName is the name of a class in networks.py, dims is a comma-separated list of integers, 
    * can be anything, and .pkl is literal e.g. HebbNet[25,50,1]_R=1.pkl"""    
    folder, filename = os.path.split(fname)
    
    if NetClass is None:
        idx = filename.find('[')
        if idx == -1:
            idx = filename.find('_')
        if idx == -1:
            raise ValueError('Invalid filename')            
        NetClass = getattr(importlib.import_module('networks'), filename[:idx])

    if dims is None:
        dims = filename[filename.find('[')+1:filename.find(']')]
        dims = [int(d) for d in dims.split(',')]
    
    if NetClass.__name__ == 'HebbDiags':
        idx = filename.find('Ng=')+3
        Ng = ''
        while filename[idx].isdigit():
            Ng += filename[idx]
            idx += 1
        net = NetClass([dims, int(Ng)])
    elif NetClass.__name__ == 'HebbFeatureLayer':
        idx = filename.find('_Nx=')+4
        Nx = ''
        while filename[idx].isdigit():
            Nx += filename[idx]
            idx += 1
        net = NetClass(dims, Nx=int(Nx))
    else:
        net = NetClass(dims)
    
    net.load(fname)
    return net    

            



        

    
