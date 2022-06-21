import itertools
import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import numpy as np
import scipy.special as sps

from net_utils import StatefulBase, check_dims, random_weight_init, binary_classifier_accuracy, xe_classifier_accuracy

#%%############
### Hebbian ###
###############

class HebbNet(StatefulBase):     
    def __init__(self, init, verbose=True, **hebbArgs):        
        super(HebbNet, self).__init__()        
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            W,b = random_weight_init([Nx,Nh,Ny], bias=True)
            # For initial state
            W2, _ = random_weight_init([Nx,Nh], bias=False)
            
            self.n_inputs = Nx
            self.n_hidden = Nh
            self.n_outputs = Ny 
        else:
            W,b = init
#            check_dims(W,b)
        
        self.verbose = verbose
        init_string = 'Net parameters:'

        self.stpType = hebbArgs.pop('stpType', 'add')
        if self.stpType == 'add':
            init_string += '\n  Net Type: Additive'
        elif self.stpType == 'mult':
            init_string += '\n  Net Type: Multiplicative'
        else:
            raise ValueError('stpType not recognized')

        self.loss_fn = F.cross_entropy # Reductions is mean by default

        self.acc_fn = xe_classifier_accuracy 

        self.fAct = hebbArgs.pop('fAct', 'sigmoid') # hidden activation
        if self.fAct  == 'linear':
            self.f = None
        elif self.fAct  == 'sigmoid':
            self.f = torch.sigmoid
        elif self.fAct  == 'tanh':
            self.f = torch.tanh
        elif self.fAct  == 'relu':
            self.f = torch.relu
        else:
            raise ValueError('f activaiton not recognized')

        self.fOutAct = hebbArgs.pop('fOutAct', 'linear') # output activiation   
        if self.fOutAct == 'linear':
            self.fOut = None # No activiation function for output
        else:
            raise ValueError('f0 activaiton not recognized')
        init_string += '\n  Structure: f: {} // fOut: {} // (Nx, Nh, Ny) = ({}, {}, {})'.format(
            self.fAct, self.fOutAct, self.n_inputs, self.n_hidden, self.n_outputs)

        # Determines whether or not input layer is a trainable parameter
        self.freezeInputs = hebbArgs.pop('freezeInputs', False)
        if self.freezeInputs: # Does not train input layer 
            init_string += '\n  Winp: Frozen // '
            self.register_buffer('w1', torch.tensor(W[0], dtype=torch.float))
        else:
            init_string += '\n  '
            self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        # Readout layer (always trainable)
        self.w2 = nn.Parameter(torch.tensor(W[1], dtype=torch.float))

        # Determines if hidden bias is present and trainable (can overwhelm noise during delay).
        self.hiddenBias = hebbArgs.pop('hiddenBias', True)
        if self.hiddenBias and not self.freezeInputs:
            init_string += 'Hidden bias: trainable // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        elif self.hiddenBias and self.freezeInputs:
            init_string += 'Hidden bias: frozen // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else: # No hidden bias
            init_string += 'No hidden bias // '
            self.register_buffer('b1', torch.zeros_like(torch.tensor(b[0], dtype=torch.float)))
        
        # Determines if readout bias is trinable or simply not used (easier interpretting readouts in the latter)
        self.roBias = hebbArgs.pop('roBias', True)
        if self.roBias:
            init_string += 'Readout bias: trainable // '
            self.b2 = nn.Parameter(torch.tensor(b[1], dtype=torch.float))
        else:
            init_string += 'No readout bias // '
            self.register_buffer('b2', torch.zeros_like(torch.tensor(b[1])))

        # Sparisfies the networks by creating masks
        self.sparsification = hebbArgs.pop('sparsification', 0.0)
        if self.sparsification > 0:
            self.register_buffer('w1Mask', torch.bernoulli((1-self.sparsification)*torch.ones_like(self.w1)))
        init_string += 'Sparsification: {:.2f} // '.format(self.sparsification)

        # Injects noise into the network
        self.noiseType = hebbArgs.pop('noiseType', None)
        self.noiseScale = hebbArgs.pop('noiseScale', 0.0)
        if self.noiseType is None:
            init_string += 'Noise type: None'
        elif self.noiseType in ('input', 'hidden',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)
        else:
            raise ValueError('Noise type not recognized.')

        ###### A-related specs ########
        init_string += '\n  A parameters:'

        # Change the type of update
        self.updateType = hebbArgs.pop('updateType', 'hebb')
        if self.updateType == 'oja':
            init_string += '\n    A update: Oja-like // '
        elif self.updateType == 'hebb_norm':
            init_string += '\n    A update: Normalized // '
        elif self.updateType == 'hebb':
            init_string += '\n    A update: Hebbian // '
        else:
            raise ValueError('updateType: {} not recognized'.format(self.updateType))

        # Change the type of hebbian update
        self.hebbType = hebbArgs.pop('hebbType', 'inputOutput')
        if self.hebbType == 'input':
            init_string += 'Pre-Syn Only // '
        elif self.hebbType == 'output':
            init_string += 'Post-Syn Only // '
        elif self.hebbType != 'inputOutput':
            raise ValueError('hebbType: {} not recognized'.format(self.hebbType))

        # Activation for the A update
        self.AAct = hebbArgs.pop('AAct', None)
        if self.AAct == 'tanh':
            init_string += 'A Act: tanh // '
        elif self.AAct is None:
            init_string += 'A Act: linear // '
        else:
            raise ValueError('A activation not recognized')

        # Change the type of eta parameter
        self.etaType = hebbArgs.pop('etaType', 'scalar')
        if self.etaType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Eta: {} // '.format(self.etaType)
        else:
            raise ValueError('etaType: {} not recognized'.format(self.etaType))

        # Change the type of lambda parameter
        self.lamType = hebbArgs.pop('lamType', 'scalar')
        if self.lamType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Lam: {} // '.format(self.lamType)
        else:
            raise ValueError('lamType: {} not recognized'.format(self.lamType))

        # Maximum lambda values
        self.lamClamp = hebbArgs.pop('lamClamp', 1.0)
        init_string += 'Lambda_max: {:.2f}'.format(self.lamClamp)

        # Initial A matrix of the network
        self.trainableState0 = hebbArgs.pop('trainableState0', False)        
        if self.trainableState0:
            init_string += '\n    A0: trainable'
            self.A0 = nn.Parameter(torch.tensor(W2[0], dtype=torch.float))
        else:
            init_string += '\n    A0: zeros'            
            self.register_buffer('A0', torch.zeros_like(torch.tensor(W2[0], dtype=torch.float)))

        if self.verbose: # Full summary of network parameters
            print(init_string)

        # Register_buffer                     
        self.register_buffer('A', None) 
        try:          
            self.reset_state() # Sets Hebbian weights to zeros
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in HebbNet.__init__'.format(e))
        
        self.register_buffer('plastic', torch.tensor(True))    
        self.register_buffer('forceAnti', torch.tensor(False))        # Forces eta to be negative
        self.register_buffer('forceHebb', torch.tensor(False))        # Forces eta to be positive
        self.register_buffer('groundTruthPlast', torch.tensor(False))
        
        self.init_hebb(**hebbArgs) #parameters of Hebbian rule

        ####      
        
    def load(self, filename):
        super(HebbNet, self).load(filename)
        # self.update_hebb(torch.tensor([0.]),torch.tensor([0.])) # to get self.eta right if forceHebb/forceAnti used        
    
    def reset_state(self, batchSize=1):

        self.A = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Nh,Nx]   
        self.A = self.A * self.A0.unsqueeze(0) # (B, Nh, Nx) x (1, Nh, Nx)

    def init_hebb(self):

        # Initialize different forms of eta parameter
        if self.etaType == 'scalar':
            _, b_eta = random_weight_init((1, 1), bias=True) # Xavier init between [-sqrt(3), sqrt(3)]
            eta = [[[b_eta[0][0]]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.etaType == 'pre_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_inputs], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.etaType == 'post_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)   
        elif self.etaType == 'matrix':
            w_eta, _ = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
            eta = w_eta[0]
            eta = eta[np.newaxis, :, :] # makes (1, Nh, Nx)  

        # Initialize different forms of lambda parameter
        if self.lamType == 'scalar': # For scalar and scalar only, lam is automatically initialized to the clamped value, otherwise uniform
            lam = [[[self.lamClamp]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.lamType == 'pre_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs,))
            lam = lam[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.lamType == 'post_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_hidden,))
            lam = lam[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)   
        elif self.lamType == 'matrix':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs, self.n_hidden,))
            lam = lam[np.newaxis, :, :] # makes (1, Nh, Nx) 
            
        #Hebbian learning rate 
        if self.forceAnti:
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = torch.exp(self._eta)
        else: # Unconstrained eta
            if self.freezeInputs:
                if self.etaType != 'scalar':
                    raise NotImplementedError('Still need to set defaults for non-scalar eta')
                # self.register_buffer('_eta', torch.tensor(-1.0, dtype=torch.float)) # Anti-hebbian
                self.register_buffer('_eta', torch.tensor(1.0, dtype=torch.float))
                self.eta = self._eta
            else:
                self._eta = nn.Parameter(torch.tensor(eta, dtype=torch.float))    
                self.eta = self._eta.data    

        # Setting lambda parameter
        if self.freezeInputs:
            if self.lamType != 'scalar':
                raise NotImplementedError('Still need to set defaults for non-scalar lambda')
            self.register_buffer('_lam', torch.tensor(self.lamClamp))
            self.lam = self._lam
        else:
            self._lam = nn.Parameter(torch.tensor(lam, dtype=torch.float))
            self.lam = self._lam.data
    
    
    def update_hebb(self, pre, post, isFam=False, stateOnly=False):
        # if self.reparamLam:
        #     self.lam = torch.sigmoid(self._lam)
        # elif self.reparamLamTau:
        #     self.lam = 1. - 1/self._lam
        # else:
        self._lam.data = torch.clamp(self._lam.data, min=0., max=self.lamClamp) 
        self.lam = self._lam 
        
        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
        
        # Changes to post and pre if ignoring respective neurons for update
        if self.hebbType == 'input':
            post = torch.ones_like(post)
        elif self.hebbType == 'output':
            pre = torch.ones_like(pre)

        if self.plastic: 
            if self.groundTruthPlast: #and isFam: # only decays hebbian weights 
                raise NotImplementedError('Broke this functionality to get around something earlier.')
                A = self.lam*self.A
            elif self.updateType == 'hebb': # normal hebbian update
                if self.AAct is None:
                    A = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1))
                elif self.AAct == 'tanh':
                    A = torch.tanh(self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1))) # [B, Nh, 1] x [B, 1, Nx] = [B, Nh, Nx]
            elif self.updateType == 'hebb_norm': # normal hebbian update
                A_tilde = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1)) # Normalizes over input dimension 
                A = A_tilde / torch.norm(A_tilde, dim=2, keepdim=True) # [B, Nh, Nx] / [B, Nh, 1]
            elif self.updateType == 'oja': # For small eta, effectively normalizes A matrix.
                A = self.lam*self.A + self.eta*(torch.bmm(post, pre.unsqueeze(1)) + post**2 * self.A) # [B, Nh, 1] x [B, 1, Nx] + [B, Nh, 1] * [B, Nh, Nx]
                
            if stateOnly: # This option is for functionality for finding fixed points
                return A
            else:
                self.A = A
                return None

    def forward(self, x, isFam=False, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('A', self.A.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Adds noise to the input
        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x + batch_noise 

        if self.sparsification > 0: # Applies masking of weights to sparsify network
            if self.stpType == 'add':
                a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*(self.w1+self.A), x.unsqueeze(2))
            elif self.stpType == 'mult':
                a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*self.w1*(
                    self.A + torch.ones_like(self.A)), x.unsqueeze(2))
        else: # No masking of weights needed
            if self.stpType == 'add':
                a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1+self.A, x.unsqueeze(2))
            elif self.stpType == 'mult':
                a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1*(
                    self.A + torch.ones_like(self.A)), x.unsqueeze(2))
                # a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1*self.A, x.unsqueeze(2))

        # Adds noise to the preactivation
        if self.noiseType in ('hidden',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(a1), 1/np.sqrt(self.n_hidden)*torch.ones_like(a1)
                )
            a1 = a1 + batch_noise 

        h = self.f(a1) if self.f is not None else a1
        # Return is only used if finding fixed points
        A = self.update_hebb(x, h, isFam=isFam, stateOnly=stateOnly)        
        if stateOnly:
            return A

        if self.w2.numel()==1: # handles case where w2 has all the same elements
            w2 = self.w2.expand(1,h.shape[0])
        else:
            w2 = self.w2

        # (Ny) + (B, Nh) x (Nh, Ny) = (B, Ny)
        # print('h', h.shape)
        # print('b2', self.b2.shape)
        # print('w2', self.w2.shape)

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        a2 = self.b2.unsqueeze(0) + torch.mm(h.squeeze(dim=2), torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(a2) if self.fOut is not None else a2  
                           
        if debug: # Make a1 and h size [B, Nh]
            return a1.squeeze(dim=2), h.squeeze(dim=2), a2, y, self.A 
        return y   
     
    def evaluate(self, batch):
        self.reset_state(batchSize=batch[0].shape[0])
        # Output size can differ from batch[1] size because XE loss is now used 
        out_size = torch.Size([batch[1].shape[0], batch[1].shape[1], self.n_outputs]) # [B, T, Ny]

        # print('out_size', out_size)
        out = torch.empty(out_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)
        # out = torch.empty_like(batch[1]) 
        for time_idx in range(batch[0].shape[1]):

            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :] # [B, Ny]
            # print('x shape', x.shape) 
            # print('y shape', y.shape)
            out[:, time_idx] = self(x, isFam=False)
            # out[t] = self(x, isFam=bool(y)) 
            # print('out[t] shape', out[t].shape)
        return out
        
    
    @torch.no_grad()    
    def evaluate_debug(self, batch, batchMask=None, recogOnly=False, acc=True, reset=True):
        """ It's possible to make this network perform other tasks simultaneously by adding an extra output unit
        and (if necessary) overriding the loss and accuracy functions. To evaluate such a network on only the 
        recognition part of the task, set recogOnly=True
        """
        B = batch[0].shape[0]

        if reset:
            self.reset_state(batchSize=B)

        Nh,Nx = self.w1.shape
        Ny,Nh = self.w2.shape 
        T = batch[1].shape[1]
        db = {'x' : torch.empty(B,T,Nx),
              'a1' : torch.empty(B,T,Nh),
              'h' : torch.empty(B,T,Nh),
              'Wxb' : torch.empty(B,T,Nh),
              'A': torch.empty(B,T,Nh,Nx),
              'Ax' : torch.empty(B,T,Nh),
              'a2' : torch.empty(B,T,Ny),
              'out' : torch.empty(B,T,Ny),
              }
        for time_idx in range(batch[0].shape[1]):
            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :]

            db['x'][:,time_idx,:] = x
            # # Note A for this given time_idx was updated on the previous pass (so A[time_idx=0] will be A0)
            # db['A'][:,time_idx,:] = self.A
            db['Ax'][:,time_idx,:] = torch.bmm(self.A, x.unsqueeze(2)).squeeze(2) 

            db['a1'][:,time_idx], db['h'][:,time_idx,:], db['a2'][:,time_idx,:], db['out'][:,time_idx,:], db['A'][:,time_idx,:] = self(x, isFam=False, debug=True)      
            w1 = self.g1*self.w1 if hasattr(self, 'g1') and not torch.isnan(self.g1) else self.w1

            db['Wxb'][:,time_idx] = self.b1.unsqueeze(0) + torch.mm(x, torch.transpose(self.w1, 0, 1))
        
        if acc:
            db['acc'] = self.accuracy(batch, out=db['out'].to(self.w1.device), outputMask=batchMask).item()  
                        
        # if recogOnly and len(db['out'].shape)>1:
        #     db['data'] = TensorDataset(batch[0], batch[1][:,0].unsqueeze(1))
        #     db['out'] = db['out'][:,0].unsqueeze(1)
        #     db['a2'] = db['a2'][:,0].unsqueeze(1)
        #     db['acc'] = self.accuracy(batch).item()        
        return db
    
    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None, trainOutputMaskBatch=None, validOutputMask=None):
        if self.hist is None: # Initialize self.hist if needed
            self.hist = {}
        if 'eta' not in self.hist: # Adds additional quantities specific to MPN to history tracking
            self.hist['eta'] = []
            self.hist['lam'] = []

        super(HebbNet, self)._monitor_init(trainBatch, validBatch, trainOutputMaskBatch=trainOutputMaskBatch, 
                                           validOutputMask=validOutputMask)

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None, trainOutputMaskBatch=None, validOutputMask=None, runValid=False):
        super(HebbNet, self)._monitor(trainBatch, validBatch, out=out, loss=loss, acc=acc, 
                                      trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask, runValid=runValid)
                                         
        if self.hist['iter']%self.mointorFreq == 0 or runValid: 
            self.hist['eta'].append(self._eta.data.cpu().numpy())
            self.hist['lam'].append(self._lam.data.cpu().numpy()) 
            
            if hasattr(self, 'writer'):    
                self.writer.add_scalar('params/lambda', self.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta',    self.eta, self.hist['iter'])       

                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1', self.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class MultiPlasticLayer(nn.Module):  
    """
    Layer with multiplasticity

    """
    def __init__(self, init, verbose=True, **mpnArgs):
        super().__init__()   

        if all([type(x)==int for x in init]) and len(init) == 2:
            Nx, Ny = init
            W, b = random_weight_init([Nx,Ny], bias=True)
            
            self.n_inputs = Nx
            self.n_outputs = Ny 
        else:
            raise ValueError('Input format not recognized')

        self.verbose = verbose
        init_string = 'MP Layer parameters:'

        self.layerAct = mpnArgs.get('layerAct', 'sigmoid') # layer activation
        if self.layerAct  == 'linear':
            self.f = None
        elif self.layerAct  == 'sigmoid':
            self.f = torch.sigmoid
        elif self.layerAct  == 'tanh':
            self.f = torch.tanh
        elif self.layerAct  == 'relu':
            self.f = torch.relu
        else:
            raise ValueError('f activaiton not recognized')

        self.mpType = mpnArgs.get('mpType', 'add')
        if self.mpType == 'add':
            init_string += '\n  MP Type: Additive //'
        elif self.mpType == 'mult':
            init_string += '\n  MP Type: Multiplicative //'
        else:
            raise ValueError('mpnType not recognized')

        init_string += ' Activation: {} // (Nx, Ny) = ({}, {})'.format(
            self.layerAct, self.n_inputs, self.n_outputs)

        # Forces all weights belonging to input neurons to be the same sign (80/20 split for +/-)
        self.useCellTypes = mpnArgs.get('useCellTypes', False)
        if self.useCellTypes:
            # Generates random vector of 1s and -1s with expected 80/20 split, rep Exc/Inh cells
            # note this starts as a numpy array b/c W[0] is a numpy array, but after first use converted to tensor
            cellTypes_np = 2*(np.random.rand(1, self.n_inputs) > 0.2)-1 # dim (1, Nx)
            # Adjusts initial W to match allowed signs
            W[0] = np.abs(W[0]) * cellTypes_np
            self.register_buffer('cellTypes', torch.tensor(cellTypes_np, dtype=torch.int32))
            init_string += '\n  W: Exc-Inh // '
        else:
            init_string += '\n  '

        # Determines whether or not layer weights are trainable parameters
        self.freezeLayer = mpnArgs.get('freezeLayer', False)
        if self.freezeLayer: # Does not train input layer 
            init_string += 'W: Frozen // '
            self.register_buffer('w1', torch.tensor(W[0], dtype=torch.float))
        else:
            self.w1 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        
        # Determines if layer bias is present and trainable (can overwhelm noise during delay).
        self.layerBias = mpnArgs.get('layerBias', True)
        if self.layerBias and not self.freezeLayer:
            init_string += 'Layer bias: trainable // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        elif self.layerBias and self.freezeLayer:
            init_string += 'Layer bias: frozen // '
            self.b1 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else: # No hidden bias
            init_string += 'No layer bias // '
            self.register_buffer('b1', torch.zeros_like(torch.tensor(b[0], dtype=torch.float)))

        # Sparisfies the layer by creating masks for weights
        self.sparsification = mpnArgs.get('sparsification', 0.0)
        if self.sparsification > 0:
            self.register_buffer('w1Mask', torch.bernoulli((1-self.sparsification)*torch.ones_like(self.w1)))
        init_string += 'Sparsification: {:.2f} // '.format(self.sparsification)

        # Injects noise into the layer
        self.noiseType = mpnArgs.get('noiseType', None)
        self.noiseScale = mpnArgs.get('noiseScale', 0.0)
        if self.noiseType in ('layer',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)
        else:
            init_string += 'Layer Noise: None'

        ###### SM matrix-related specs ########
        init_string += '\n  SM matrix parameters:'

        # Change the type of update
        self.updateType = mpnArgs.get('updateType', 'hebb')
        if self.updateType == 'oja':
            init_string += '\n    M update: Oja-like // '
        elif self.updateType == 'hebb_norm':
            init_string += '\n    M update: Normalized // '
        elif self.updateType == 'hebb':
            init_string += '\n    M update: Hebbian // '
        else:
            raise ValueError('updateType: {} not recognized'.format(self.updateType))

        # Change the type of hebbian update
        self.hebbType = mpnArgs.get('hebbType', 'inputOutput')
        if self.hebbType == 'input':
            init_string += 'Pre-Syn Only // '
        elif self.hebbType == 'output':
            init_string += 'Post-Syn Only // '
        elif self.hebbType != 'inputOutput':
            raise ValueError('hebbType: {} not recognized'.format(self.hebbType))

        # Activation for the M update
        self.MAct = mpnArgs.get('MAct', None)
        if self.MAct == 'tanh':
            init_string += 'M Act: tanh // '
        elif self.MAct is None:
            init_string += 'M Act: linear // '
        else:
            raise ValueError('M activation not recognized')

        # Initial SM matrix of the network
        self.trainableState0 = mpnArgs.get('trainableState0', False)
        W2, _ = random_weight_init([self.n_inputs, self.n_outputs], bias=False)        
        if self.trainableState0:
            init_string += 'M0: trainable'
            self.M0 = nn.Parameter(torch.tensor(W2[0], dtype=torch.float))
        else:
            init_string += 'M0: zeros'            
            self.register_buffer('M0', torch.zeros_like(torch.tensor(W2[0], dtype=torch.float)))

        # Change the type of eta parameter
        self.etaType = mpnArgs.get('etaType', 'scalar')
        if self.etaType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += '\n    Eta: {} // '.format(self.etaType)
        else:
            raise ValueError('etaType: {} not recognized'.format(self.etaType))

        # Change the type of lambda parameter
        self.lamType = mpnArgs.get('lamType', 'scalar')
        if self.lamType in ('scalar', 'pre_vector', 'post_vector', 'matrix'):
            init_string += 'Lam: {} // '.format(self.lamType)
        else:
            raise ValueError('lamType: {} not recognized'.format(self.lamType))

        # Maximum lambda values
        self.lamClamp = mpnArgs.get('lamClamp', 1.0)
        init_string += 'Lambda_max: {:.2f}'.format(self.lamClamp)

        if self.verbose: # Full summary of network parameters
            print(init_string)

        # Register_buffer                     
        self.register_buffer('M', None) 
        try:          
            self.reset_state() # Sets Hebbian weights to zeros
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in mpnNet.__init__'.format(e))
        
        self.register_buffer('plastic', torch.tensor(True))    
        self.register_buffer('forceAnti', torch.tensor(False))        # Forces eta to be negative
        self.register_buffer('forceHebb', torch.tensor(False))        # Forces eta to be positive
        self.register_buffer('groundTruthPlast', torch.tensor(False))
        
        self.init_sm_matrix()

    def reset_state(self, batchSize=1):

        self.M = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Ny,Nx]   
        self.M = self.M * self.M0.unsqueeze(0) # (B, Ny, Nx) x (1, Ny, Nx)

    def init_sm_matrix(self):

        # Initialize different forms of eta parameter
        if self.etaType == 'scalar':
            _, b_eta = random_weight_init((1, 1), bias=True) # Xavier init between [-sqrt(3), sqrt(3)]
            eta = [[[b_eta[0][0]]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.etaType == 'pre_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_inputs], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.etaType == 'post_vector':
            _, b_eta = random_weight_init([self.n_inputs, self.n_outputs], bias=True)
            eta = b_eta[0]
            eta = eta[np.newaxis, :, np.newaxis] # makes (1, Ny, 1)   
        elif self.etaType == 'matrix':
            w_eta, _ = random_weight_init([self.n_inputs, self.n_outputs], bias=True)
            eta = w_eta[0]
            eta = eta[np.newaxis, :, :] # makes (1, Ny, Nx)  

        # Initialize different forms of lambda parameter
        if self.lamType == 'scalar': # For scalar and scalar only, lam is automatically initialized to the clamped value, otherwise uniform
            lam = [[[self.lamClamp]]] # shape (1, 1, 1)
            # eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, shape (1, 1, 1)
        elif self.lamType == 'pre_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs,))
            lam = lam[np.newaxis, np.newaxis, :] # makes (1, 1, Nx)
        elif self.lamType == 'post_vector':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_outputs,))
            lam = lam[np.newaxis, :, np.newaxis] # makes (1, Ny, 1)   
        elif self.lamType == 'matrix':
            lam = np.random.uniform(low=0.0, high=self.lamClamp, size=(self.n_inputs, self.n_outputs,))
            lam = lam[np.newaxis, :, :] # makes (1, Ny, Nx) 
            
        #Hebbian learning rate 
        if self.forceAnti:
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            if self.eta: 
                del self.eta
            self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            self.eta = torch.exp(self._eta)
        else: # Unconstrained eta
            if self.freezeLayer:
                if self.etaType != 'scalar':
                    raise NotImplementedError('Still need to set defaults for non-scalar eta')
                # self.register_buffer('_eta', torch.tensor(-1.0, dtype=torch.float)) # Anti-hebbian
                self.register_buffer('_eta', torch.tensor(1.0, dtype=torch.float))
                self.eta = self._eta
            else:
                self._eta = nn.Parameter(torch.tensor(eta, dtype=torch.float))    
                self.eta = self._eta.data    

        # Setting lambda parameter
        if self.freezeLayer:
            if self.lamType != 'scalar':
                raise NotImplementedError('Still need to set defaults for non-scalar lambda')
            self.register_buffer('_lam', torch.tensor(self.lamClamp))
            self.lam = self._lam
        else:
            self._lam = nn.Parameter(torch.tensor(lam, dtype=torch.float))
            self.lam = self._lam.data

    def update_sm_matrix(self, pre, post, stateOnly=False):
        """
        Updates the synaptic modulation matrix from one time step to the next.
        Should only be called in the forward pass once.

        Note that this directly updates self.M, unless stateOnly=True in which case
        it returns the updated state.

        """

        self._lam.data = torch.clamp(self._lam.data, min=0., max=self.lamClamp) 
        self.lam = self._lam 
        
        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
        
        # Changes to post and pre if ignoring respective neurons for update
        if self.hebbType == 'input':
            post = torch.ones_like(post)
        elif self.hebbType == 'output':
            pre = torch.ones_like(pre)

        if self.plastic: 
            if self.groundTruthPlast: #and isFam: # only decays hebbian weights 
                raise NotImplementedError('Broke this functionality to get around something earlier.')
                M = self.lam*self.M
            elif self.updateType == 'hebb': # normal hebbian update
                if self.MAct is None:
                    M = self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1))
                elif self.MAct == 'tanh':
                    M = torch.tanh(self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1))) # [B, Ny, 1] x [B, 1, Nx] = [B, Ny, Nx]
            elif self.updateType == 'hebb_norm': # normal hebbian update
                M_tilde = self.lam*self.M + self.eta*torch.bmm(post, pre.unsqueeze(1)) # Normalizes over input dimension 
                M = M_tilde / torch.norm(M_tilde, dim=2, keepdim=True) # [B, Ny, Nx] / [B, Ny, 1]
            elif self.updateType == 'oja': # For small eta, effectively normalizes M matrix.
                M = self.lam*self.M + self.eta*(torch.bmm(post, pre.unsqueeze(1)) + post**2 * self.M) # [B, Ny, 1] x [B, 1, Nx] + [B, Ny, 1] * [B, Ny, Nx]
                
            if stateOnly: # This option is for functionality for finding fixed points
                return M
            else:
                self.M = M
                return None

    def forward(self, x, debug=False, stateOnly=False):
        """
        Passes inputs through the network and also modifies the internal state of the model (self.M). 
        Don't call twice in a row unless you want to update self.M twice!

        x.shape: [B, Nx]
        b1.shape: [Ny]
        w1.shape=[Ny,Nx], 
        M.shape=[B,Ny,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('M', self.M.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Clamps w1 before it is used in forward pass if cellTypes are being used
        if self.useCellTypes: # First multiplication removes signs, then clamps, then restores them
            self.w1.data = torch.clamp(self.w1.data*self.cellTypes, min=0., max=1e6) * self.cellTypes

        if self.sparsification == 0.0:
            if self.mpType == 'add':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1+self.M, x.unsqueeze(2))
            elif self.mpType == 'mult':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1*(
                        self.M + torch.ones_like(self.M)), x.unsqueeze(2))
                # y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1*self.M, x.unsqueeze(2))
        else: # Applies masking of weights to sparsify network
            if self.mpType == 'add':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*(self.w1+self.M), x.unsqueeze(2))
            elif self.mpType == 'mult':
                y_tilde = torch.baddbmm(self.b1.unsqueeze(1), self.w1Mask*self.w1*(
                        self.M + torch.ones_like(self.M)), x.unsqueeze(2))

        # Adds noise to the preactivations
        if self.noiseType in ('layer',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(y_tilde), 1/np.sqrt(self.n_outputs)*torch.ones_like(y_tilde)
                )
            y_tilde = y_tilde + batch_noise 

        y = self.f(y_tilde) if self.f is not None else y_tilde
        
        M = self.update_sm_matrix(x, y, stateOnly=stateOnly) # Returned M is only used if finding fixed points        
        
        if stateOnly:
            return M                 
        elif debug:
            return y_tilde, y, self.M
        else:
            return y  

class MultiPlasticNet(StatefulBase):     
    def __init__(self, init, verbose=True, **mpnArgs):        
        super(MultiPlasticNet, self).__init__()        
        
        if all([type(x)==int for x in init]) and len(init) == 3:
            Nx,Nh,Ny = init
            # For readouts
            W,b = random_weight_init([Nh,Ny], bias=True)
            
            self.n_inputs = Nx
            self.n_hidden = Nh
            self.n_outputs = Ny 
        else:
            raise ValueError('Init type not recognized.')
        
        self.verbose = verbose

        self.loss_fn = F.cross_entropy # Reductions is mean by default
        self.acc_fn = xe_classifier_accuracy 

        # Creates the input MP layer
        self.mp_layer = MultiPlasticLayer((self.n_inputs, self.n_hidden), verbose=verbose, **mpnArgs)

        init_string = 'Network parameters:'

        self.fOutAct = mpnArgs.pop('fOutAct', 'linear') # output activiation   
        if self.fOutAct == 'linear':
            self.fOut = None # No activiation function for output
        else:
            raise ValueError('f0 activaiton not recognized')
        
        init_string += '\n  Readout act: {} // '.format(self.fOutAct)

        # Readout layer (always trainable)
        self.w2 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        
        # Determines if readout bias is trainable or simply not used (easier interpretting readouts in the latter)
        self.roBias = mpnArgs.pop('roBias', True)
        if self.roBias:
            init_string += 'Readout bias: trainable // '
            self.b2 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else:
            init_string += 'No readout bias // '
            self.register_buffer('b2', torch.zeros_like(torch.tensor(b[0])))

        # Injects noise into the network
        self.noiseType = mpnArgs.get('noiseType', None)
        self.noiseScale = mpnArgs.get('noiseScale', 0.0)
        if self.noiseType is None:
            init_string += 'Noise type: None'
        elif self.noiseType in ('input',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)

        if self.verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)      
        
    def load(self, filename):
        super(MultiPlasticNet, self).load(filename)
        # self.update_hebb(torch.tensor([0.]),torch.tensor([0.])) # to get self.eta right if forceHebb/forceAnti used        
    
    def reset_state(self, batchSize=1):
        """
        Resets states of all internal layer SM matrices
        """
        self.mp_layer.reset_state(batchSize=batchSize)   

    def forward(self, x, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.M). 
        Don't call twice in a row unless you want to update self.M twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('M', self.M.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Adds noise to the input
        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x + batch_noise 

        outs = self.mp_layer(x, debug=debug, stateOnly=stateOnly)

        if stateOnly:
            M = outs
            return M
        elif debug:
            h_tilde, h, M = outs
        else:
            h = outs
        # (Ny) + (B, Nh) x (Nh, Ny) = (B, Ny)
        # print('h', h.shape)
        # print('b2', self.b2.shape)
        # print('w2', self.w2.shape)

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        y_tilde = self.b2.unsqueeze(0) + torch.mm(h.squeeze(dim=2), torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(y_tilde) if self.fOut is not None else y_tilde  
                           
        if debug: # Make a1 and h size [B, Nh]
            return h_tilde.squeeze(dim=2), h.squeeze(dim=2), y_tilde, y, M 
        else:
            return y   
     
    def evaluate(self, batch):
        """
        Runs a full sequence of the given back size through the network.
        """
        self.reset_state(batchSize=batch[0].shape[0])
        # Output size can differ from batch[1] size because XE loss is now used 
        out_size = torch.Size([batch[1].shape[0], batch[1].shape[1], self.n_outputs]) # [B, T, Ny]

        # print('out_size', out_size)
        out = torch.empty(out_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)
        # out = torch.empty_like(batch[1]) 
        for time_idx in range(batch[0].shape[1]):

            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :] # [B, Ny]
            # print('x shape', x.shape) 
            # print('y shape', y.shape)
            out[:, time_idx] = self(x)
            # print('out[t] shape', out[t].shape)
        return out
        
    @torch.no_grad()    
    def evaluate_debug(self, batch, batchMask=None, acc=True, reset=True):
        """ 
        Runs a full sequence of the given back size through the network, but now keeps track of all sorts of parameters
        """
        B = batch[0].shape[0]

        if reset:
            self.reset_state(batchSize=B)

        Nx = self.n_inputs
        Nh = self.n_hidden
        Ny = self.n_outputs
        T = batch[1].shape[1]
        db = {'x' : torch.empty(B,T,Nx),
              'h_tilde' : torch.empty(B,T,Nh),
              'h' : torch.empty(B,T,Nh),
              'Wxb' : torch.empty(B,T,Nh),
              'M': torch.empty(B,T,Nh,Nx),
              'Mx' : torch.empty(B,T,Nh),
              'y_tilde' : torch.empty(B,T,Ny),
              'out' : torch.empty(B,T,Ny),
              }
        for time_idx in range(batch[0].shape[1]):
            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :]

            db['x'][:,time_idx,:] = x
            # # Note A for this given time_idx was updated on the previous pass (so A[time_idx=0] will be A0)
            (db['h_tilde'][:,time_idx], db['h'][:,time_idx,:], db['y_tilde'][:,time_idx,:], 
                db['out'][:,time_idx,:], db['M'][:,time_idx,:]) = self(x, debug=True)      
            db['Mx'][:,time_idx,:] = torch.bmm(self.mp_layer.M, x.unsqueeze(2)).squeeze(2) 

            db['Wxb'][:,time_idx] = self.mp_layer.b1.unsqueeze(0) + torch.mm(x, torch.transpose(self.mp_layer.w1, 0, 1))
        
        if acc:
            db['acc'] = self.accuracy(batch, out=db['out'].to(self.w2.device), outputMask=batchMask).item()  
                             
        return db
    
    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None, trainOutputMaskBatch=None, validOutputMask=None):
        if self.hist is None: # Initialize self.hist if needed
            self.hist = {}
        if 'eta' not in self.hist: # Adds additional quantities specific to MPN to history tracking
            self.hist['eta'] = []
            self.hist['lam'] = []

        super(MultiPlasticNet, self)._monitor_init(trainBatch, validBatch, trainOutputMaskBatch=trainOutputMaskBatch, 
                                           validOutputMask=validOutputMask)

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None, trainOutputMaskBatch=None, validOutputMask=None, runValid=False):
        super(MultiPlasticNet, self)._monitor(trainBatch, validBatch, out=out, loss=loss, acc=acc, 
                                      trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask, runValid=runValid)
                                         
        if self.hist['iter']%self.mointorFreq == 0 or runValid: 
            self.hist['eta'].append(self.mp_layer._eta.data.cpu().numpy())
            self.hist['lam'].append(self.mp_layer._lam.data.cpu().numpy()) 
            
            if hasattr(self, 'writer'):    
                self.writer.add_scalar('params/lambda', self.mp_layer.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta',    self.mp_layer.eta, self.hist['iter'])       

                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1', self.mp_layer.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])


class MultiPlasticNetTwo(StatefulBase):     
    def __init__(self, init, verbose=True, **mpnArgs):        
        super(MultiPlasticNetTwo, self).__init__()        
        
        if all([type(x)==int for x in init]) and len(init) == 4:
            Nx,Nh1,Nh2,Ny = init
            # For readouts
            W,b = random_weight_init([Nh2,Ny], bias=True)
            
            self.n_inputs = Nx
            self.n_hidden1 = Nh1
            self.n_hidden2 = Nh2
            self.n_outputs = Ny 
        else:
            raise ValueError('Init type not recognized.')
        
        self.verbose = verbose

        self.loss_fn = F.cross_entropy # Reductions is mean by default
        self.acc_fn = xe_classifier_accuracy 

        # Creates the input MP layer
        self.mp_layer1 = MultiPlasticLayer((self.n_inputs, self.n_hidden1), verbose=verbose, **mpnArgs)
        self.mp_layer2 = MultiPlasticLayer((self.n_hidden1, self.n_hidden2), verbose=verbose, **mpnArgs)

        init_string = 'Network parameters:'

        self.fOutAct = mpnArgs.pop('fOutAct', 'linear') # output activiation   
        if self.fOutAct == 'linear':
            self.fOut = None # No activiation function for output
        else:
            raise ValueError('f0 activaiton not recognized')
        
        init_string += '\n  Readout act: {} // '.format(self.fOutAct)

        # Readout layer (always trainable)
        self.w2 = nn.Parameter(torch.tensor(W[0], dtype=torch.float))
        
        # Determines if readout bias is trainable or simply not used (easier interpretting readouts in the latter)
        self.roBias = mpnArgs.pop('roBias', True)
        if self.roBias:
            init_string += 'Readout bias: trainable // '
            self.b2 = nn.Parameter(torch.tensor(b[0], dtype=torch.float))
        else:
            init_string += 'No readout bias // '
            self.register_buffer('b2', torch.zeros_like(torch.tensor(b[0])))

        # Injects noise into the network
        self.noiseType = mpnArgs.get('noiseType', None)
        self.noiseScale = mpnArgs.get('noiseScale', 0.0)
        if self.noiseType is None:
            init_string += 'Noise type: None'
        elif self.noiseType in ('input',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)

        if self.verbose: # Full summary of readout parameters (MP layer prints out internally)
            print(init_string)      
        
    def load(self, filename):
        super(MultiPlasticNetTwo, self).load(filename)
        # self.update_hebb(torch.tensor([0.]),torch.tensor([0.])) # to get self.eta right if forceHebb/forceAnti used        
    
    def reset_state(self, batchSize=1):
        """
        Resets states of all internal layer SM matrices
        """
        self.mp_layer1.reset_state(batchSize=batchSize)
        self.mp_layer2.reset_state(batchSize=batchSize)    

    def forward(self, x, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.M). 
        Don't call twice in a row unless you want to update self.M twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """
                
        # w1 = self.g1*self.w1 if not torch.isnan(self.g1) else self.w1
        
        # print('M', self.M.shape)
        # print('x', x.shape)
        # print('b1', self.b1.shape)
        # print('w1', self.w1.shape)

        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)

        # Adds noise to the input
        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x + batch_noise 

        outs1 = self.mp_layer1(x, debug=debug, stateOnly=stateOnly)

        if stateOnly:
            raise NotImplementedError()
        elif debug:
            h_tilde1, h1, M1 = outs1
        else:
            h1 = outs1
        
        outs2 = self.mp_layer2(h1.squeeze(dim=2), debug=debug, stateOnly=stateOnly)

        if debug:
            h_tilde2, h2, M2 = outs2
        else:
            h2 = outs2

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        y_tilde = self.b2.unsqueeze(0) + torch.mm(h2.squeeze(dim=2), torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(y_tilde) if self.fOut is not None else y_tilde  
                           
        if debug: # Make a1 and h size [B, Nh]
            return h_tilde1.squeeze(dim=2), h1.squeeze(dim=2), h_tilde2.squeeze(dim=2), h2.squeeze(dim=2), y_tilde, y, M1, M2
        else:
            return y   
     
    def evaluate(self, batch):
        """
        Runs a full sequence of the given back size through the network.
        """
        self.reset_state(batchSize=batch[0].shape[0])
        # Output size can differ from batch[1] size because XE loss is now used 
        out_size = torch.Size([batch[1].shape[0], batch[1].shape[1], self.n_outputs]) # [B, T, Ny]

        # print('out_size', out_size)
        out = torch.empty(out_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)
        # out = torch.empty_like(batch[1]) 
        for time_idx in range(batch[0].shape[1]):

            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :] # [B, Ny]
            # print('x shape', x.shape) 
            # print('y shape', y.shape)
            out[:, time_idx] = self(x)
            # print('out[t] shape', out[t].shape)
        return out
        
    @torch.no_grad()    
    def evaluate_debug(self, batch, batchMask=None, acc=True, reset=True):
        """ 
        Runs a full sequence of the given back size through the network, but now keeps track of all sorts of parameters
        """
        B = batch[0].shape[0]

        if reset:
            self.reset_state(batchSize=B)

        Nx = self.n_inputs
        Nh1 = self.n_hidden1
        Nh2 = self.n_hidden2
        Ny = self.n_outputs
        T = batch[1].shape[1]
        db = {'x' : torch.empty(B,T,Nx),
              'h_tilde1' : torch.empty(B,T,Nh1),
              'h1' : torch.empty(B,T,Nh1),
              'M1': torch.empty(B,T,Nh1,Nx),
              'h_tilde2' : torch.empty(B,T,Nh2),
              'h2' : torch.empty(B,T,Nh2),
              'M2': torch.empty(B,T,Nh2,Nh1),
              'y_tilde' : torch.empty(B,T,Ny),
              'out' : torch.empty(B,T,Ny),
              }
        for time_idx in range(batch[0].shape[1]):
            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :]

            db['x'][:,time_idx,:] = x
            # # Note A for this given time_idx was updated on the previous pass (so A[time_idx=0] will be A0)
            (db['h_tilde1'][:,time_idx], db['h1'][:,time_idx,:], 
                db['h_tilde2'][:,time_idx], db['h2'][:,time_idx,:],
                db['y_tilde'][:,time_idx,:], db['out'][:,time_idx,:], 
                db['M1'][:,time_idx,:], db['M2'][:,time_idx,:]) = self(x, debug=True)      
            
        
        if acc:
            db['acc'] = self.accuracy(batch, out=db['out'].to(self.w2.device), outputMask=batchMask).item()  
                             
        return db
    
    @torch.no_grad()
    def _monitor_init(self, trainBatch, validBatch=None, trainOutputMaskBatch=None, validOutputMask=None):
        if self.hist is None: # Initialize self.hist if needed
            self.hist = {}
        if 'eta1' not in self.hist: # Adds additional quantities specific to MPN to history tracking
            self.hist['eta1'] = []
            self.hist['lam1'] = []
            self.hist['eta2'] = []
            self.hist['lam2'] = []

        super(MultiPlasticNetTwo, self)._monitor_init(trainBatch, validBatch, trainOutputMaskBatch=trainOutputMaskBatch, 
                                           validOutputMask=validOutputMask)

    @torch.no_grad()
    def _monitor(self, trainBatch, validBatch=None, out=None, loss=None, acc=None, trainOutputMaskBatch=None, validOutputMask=None, runValid=False):
        super(MultiPlasticNetTwo, self)._monitor(trainBatch, validBatch, out=out, loss=loss, acc=acc, 
                                      trainOutputMaskBatch=trainOutputMaskBatch, validOutputMask=validOutputMask, runValid=runValid)
                                         
        if self.hist['iter']%self.mointorFreq == 0 or runValid: 
            self.hist['eta1'].append(self.mp_layer1._eta.data.cpu().numpy())
            self.hist['lam1'].append(self.mp_layer1._lam.data.cpu().numpy()) 
            self.hist['eta2'].append(self.mp_layer2._eta.data.cpu().numpy())
            self.hist['lam2'].append(self.mp_layer2._lam.data.cpu().numpy()) 
            
            if hasattr(self, 'writer'):    
                self.writer.add_scalar('params/lambda', self.mp_layer.lam, self.hist['iter'])
                self.writer.add_scalar('params/eta',    self.mp_layer.eta, self.hist['iter'])       

                if self.w2.numel()==1:
                    self.writer.add_scalar('params/w2', self.w2.item(), self.hist['iter'])
                if self.b1.numel()==1:
                    self.writer.add_scalar('params/b1', self.mp_layer.b1.item(), self.hist['iter'])
                if self.b2.numel()==1:
                    self.writer.add_scalar('params/b2', self.b2.item(), self.hist['iter'])
                if self.g1 is not None:
                    self.writer.add_scalar('params/g1', self.g1.item(), self.hist['iter'])

class RecHebbNet(HebbNet):     
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(RecHebbNet, self).__init__(init, f=f, fOut=fOut, **hebbArgs)        
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            Wrec, _ = random_weight_init([Nx,Nh,Nh], bias=True)
            # For initial state
            W2rec, b2rec = random_weight_init([Nx,Nh,Nh], bias=False)
        else:
            W,b = init
        
        # Recurrent weights
        self.wr = nn.Parameter(torch.tensor(Wrec[1], dtype=torch.float))
        # Adds B and h as trainable initial states
        if self.trainableState0:
            self.B0 = nn.Parameter(torch.tensor(W2rec[1], dtype=torch.float))
            self.h0 = nn.Parameter(torch.tensor(b2rec[0], dtype=torch.float)) 
        else:
            if self.stpType == 'add':
                self.B0 = torch.zeros_like(torch.tensor(W2rec[1], dtype=torch.float))
            elif self.stpType == 'mult':
                self.B0 = torch.ones_like(torch.tensor(W2rec[1], dtype=torch.float))
            self.h0 = torch.zeros_like(torch.tensor(b2rec[0], dtype=torch.float))
      

        # Register_buffer                     
        self.register_buffer('A', None)
        self.register_buffer('B', None)  
        try:          
            self.reset_state() # Sets Hebbian weights to zeros
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in RecHebbNet.__init__'.format(e))
        
        self.init_hebb_rec(**hebbArgs) #parameters of Hebbian rule    
    
    def reset_state(self, batchSize=1):

        self.A = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Nh,Nx]   
        self.A = self.A * self.A0.unsqueeze(0) # (B, Nh, Nx) x (1, Nh, Nx)

        self.B = torch.ones(batchSize, *self.wr.shape, device=self.wr.device) #shape=[B,Nh,Nh]   
        self.B = self.B * self.B0.unsqueeze(0) # (B, Nh, Nh) x (1, Nh, Nh)

        self.h = torch.ones(batchSize, *self.b1.shape, device=self.b1.device) #shape=[B,Nh]  
        self.h = self.h * self.h0.unsqueeze(0) # (B, Nh) x (1, Nh)

    def init_hebb_rec(self, eta=None, lam=0.99, **hebbArgs):
        if eta is None:
            if self.etaType == 'scalar':
                eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5, (1, 1, 1)
                etar = [[[-5./self.w1.shape[1]]]]
            elif self.etaType == 'vector':
                _, b_eta = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
                eta = b_eta[0]
                eta = eta[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)
                _, b_etar = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
                etar = b_etar[0]
                etar = etar[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)

        self._lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.lam = self._lam.data
        self._lamr = nn.Parameter(torch.tensor(lam))
        self.lamr = self._lamr.data
            
        #Hebbian learning rate 
        if self.forceAnti:
            raise NotImplementedError()
            # if self.eta: 
            #     del self.eta
            # self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            # self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            raise NotImplementedError()
            # if self.eta: 
            #     del self.eta
            # self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta, dtype=torch.float)))) #eta = exp(_eta)
            # self.eta = torch.exp(self._eta)
        else:
            self._eta = nn.Parameter(torch.tensor(eta, dtype=torch.float))    
            self.eta = self._eta.data

            self._etar = nn.Parameter(torch.tensor(etar, dtype=torch.float))    
            self.etar = self._etar.data    
    
    
    def update_hebb(self, pre, post, preh, isFam=False, stateOnly=False):
        self._lam.data = torch.clamp(self._lam.data, min=0., max=self.lamClamp) #if lam>1, get exponential explosion
        self.lam = self._lam 
        self._lamr.data = torch.clamp(self._lamr.data, min=0., max=self.lamClamp) 
        self.lamr = self._lamr 
        
        if self.forceAnti:
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
            self.etar = self._etar
        
        # Changes to post and pre if ignoring respective neurons for update
        if self.hebbType == 'input':
            post = torch.ones_like(post)
        elif self.hebbType == 'output':
            pre = torch.ones_like(pre)
            preh = torch.ones_like(preh)

        if self.plastic: 
            if self.groundTruthPlast: #and isFam: # only decays hebbian weights 
                raise NotImplementedError('Broke this functionality to get around something earlier.')
                A = self.lam*self.A
            elif self.updateType == 'hebb': # normal hebbian update
                A = self.lam*self.A + self.eta*torch.bmm(post.unsqueeze(2), pre.unsqueeze(1)) # [B, Nh, 1] x [B, 1, Nx] = [B, Nh, Nx]
                B = self.lamr*self.B + self.etar*torch.bmm(post.unsqueeze(2), preh.unsqueeze(1)) # [B, Nh, 1] x [B, 1, Nh] = [B, Nh, Nh]
            elif self.updateType == 'hebb_norm': # normal hebbian update
                raise NotImplementedError()
                # A_tilde = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1)) # Normalizes over input dimension 
                # A = A_tilde / torch.norm(A_tilde, dim=2, keepdim=True) # [B, Nh, Nx] / [B, Nh, 1]
            elif self.updateType == 'oja': # For small eta, effectively normalizes A matrix.
                raise NotImplementedError()
                # A = self.lam*self.A + self.eta*(torch.bmm(post, pre.unsqueeze(1)) + post**2 * self.A) # [B, Nh, 1] x [B, 1, Nx] + [B, Nh, 1] * [B, Nh, Nx]
                
            if stateOnly: # This code is for functionality for finding fixed points
                return A, B
            else:
                self.A = A
                self.B = B
                return None, None

    def forward(self, x, isFam=False, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """

        # print('w1+A', (self.w1+self.A).shape)
        # print('wr+B', (self.wr+self.B).shape)
        # print('h', self.h.shape)

        if self.stpType == 'add':
            a1 = (torch.baddbmm(self.b1.unsqueeze(1), self.w1+self.A, x.unsqueeze(2)) # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)
                + torch.bmm(self.wr+self.B, self.h.unsqueeze(2)) # (1, Nh, Nh) x (B, Nh, Nh) = (B, Nh)
                ) 
        elif self.stpType == 'mult':
            a1 = (torch.baddbmm(self.b1.unsqueeze(1), self.w1*self.A, x.unsqueeze(2))
                + torch.bmm(self.wr*self.B, self.h.unsqueeze(2)) # (1, Nh, Nh) x (B, Nh, Nh) = (B, Nh)
                ) 
        # a1 = torch.baddbmm(self.b1.unsqueeze(1), torch.transpose(self.w1, 0, 1)+self.A, x)
        h_new = (self.f(a1) if self.f is not None else a1).squeeze(dim=2)
        # Return is only used if finding fixed points
        A, B = self.update_hebb(x, h_new, self.h, isFam=isFam, stateOnly=stateOnly)        
        if stateOnly:
            return A, B, h_new

        self.h = h_new

        # (Ny) + (B, Nh) x (Nh, Ny) = (B, Ny)
        # print('h', h.shape)
        # print('b2', self.b2.shape)
        # print('w2', self.w2.shape)

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        a2 = self.b2.unsqueeze(0) + torch.mm(self.h, torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(a2) if self.fOut is not None else a2  
                           
        if debug: # Make a1 and h size [B, Nh]
            raise NotImplementedError()
            return a1.squeeze(dim=2), h.squeeze(dim=2), a2, y 
        return y   


class GHU(HebbNet):     
    def __init__(self, init, f=torch.sigmoid, fOut=torch.sigmoid, **hebbArgs):        
        super(GHU, self).__init__(init, f=f, fOut=fOut, **hebbArgs)        
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            Wu,bu = random_weight_init([Nx,Nh,Ny], bias=True)
            Wr,br = random_weight_init([Nx,Nh,Ny], bias=True)
            # For initial state
            W2u, _ = random_weight_init([Nx,Nh], bias=False)
            W2r, _ = random_weight_init([Nx,Nh], bias=False)
        else:
            W,b = init
#            check_dims(W,b)
        
        # Additional update and reset gates
        self.wu = nn.Parameter(torch.tensor(Wu[0], dtype=torch.float))
        self.bu = nn.Parameter(torch.tensor(bu[0], dtype=torch.float))

        self.wr = nn.Parameter(torch.tensor(Wr[0], dtype=torch.float))
        self.br = nn.Parameter(torch.tensor(br[0], dtype=torch.float))
            
        if self.trainableState0:
            self.Au0 = nn.Parameter(torch.tensor(W2u[0], dtype=torch.float))
            self.Ar0 = nn.Parameter(torch.tensor(W2r[0], dtype=torch.float))
        else:
            if self.stpType == 'add':
                self.Au0 = torch.zeros_like(torch.tensor(W2u[0], dtype=torch.float))
                self.Ar0 = torch.zeros_like(torch.tensor(W2r[0], dtype=torch.float))
            elif self.stpType == 'mult':
                self.Au0 = torch.ones_like(torch.tensor(W2u[0], dtype=torch.float))
                self.Ar0 = torch.ones_like(torch.tensor(W2r[0], dtype=torch.float))                    

        # Register_buffer                     
        self.register_buffer('A', None)
        self.register_buffer('Au', None)
        self.register_buffer('Ar', None)   
        try:          
            self.reset_state() # Sets Hebbian weights to zeros
        except AttributeError as e:
            print('Warning: {}. Not running reset_state() in GHU.__init__'.format(e))


        self.init_hebb_GHU(**hebbArgs) #parameters of Hebbian rule

        ####          
    
    def reset_state(self, batchSize=1):
        self.A = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Nh,Nx]   
        self.A = self.A * self.A0.unsqueeze(0) # (B, Nh, Nx) x (1, Nh, Nx)
        
        self.Au = torch.ones(batchSize, *self.wr.shape, device=self.wr.device)  
        self.Au = self.Au * self.Au0.unsqueeze(0)

        self.Ar = torch.ones(batchSize, *self.wu.shape, device=self.wu.device)
        self.Ar = self.Ar * self.Ar0.unsqueeze(0) 


    def init_hebb_GHU(self, eta=None, lam=0.99, **hebbArgs):
        if eta is None:
            if self.etaType == 'scalar':
                eta = [[[-5./self.w1.shape[1]]]] #eta*d = -5
                etau = [[[-5./self.w1.shape[1]]]]
                etar = [[[-5./self.w1.shape[1]]]]
            elif self.etaType == 'vector':
                _, b_eta = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
                eta = b_eta[0]
                eta = eta[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)
                _, b_etau = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
                etau = b_etau[0]
                etau = etau[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)
                _, b_etar = random_weight_init([self.n_inputs, self.n_hidden], bias=True)
                etar = b_etar[0]
                etar = etar[np.newaxis, :, np.newaxis] # makes (1, Nh, 1)

        self._lam = nn.Parameter(torch.tensor(lam)) #Hebbian decay
        self.lam = self._lam.data

        self._lamu = nn.Parameter(torch.tensor(lam))
        self.lamu = self._lamu.data

        self._lamr = nn.Parameter(torch.tensor(lam))
        self.lamr = self._lamr.data
            
        #Hebbian learning rate 
        if self.forceAnti:
            raise NotImplementedError()
            # if self.eta: 
            #     del self.eta
            # self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta)))) #eta = exp(_eta)
            # self.eta = -torch.exp(self._eta)
        elif self.forceHebb:
            raise NotImplementedError() 
            # if self.eta: 
            #     del self.eta
            # self._eta = nn.Parameter(torch.log(torch.abs(torch.tensor(eta)))) #eta = exp(_eta)
            # self.eta = torch.exp(self._eta)
        else:
            self._eta = nn.Parameter(torch.tensor(eta, dtype=torch.float))    
            self.eta = self._eta.data

            self._etau = nn.Parameter(torch.tensor(etau, dtype=torch.float))    
            self.etau = self._etau.data 

            self._etar = nn.Parameter(torch.tensor(etar, dtype=torch.float))    
            self.etar = self._etar.data     
    
    
    def update_hebb(self, pre, post, postu, postr, isFam=False, stateOnly=False):
        # if self.reparamLam:
        #     self.lam = torch.sigmoid(self._lam)
        # elif self.reparamLamTau:
        #     self.lam = 1. - 1/self._lam
        # else:
        self._lam.data = torch.clamp(self._lam.data, min=0., max=self.lamClamp) #if lam>1, get exponential explosion
        self.lam = self._lam 

        self._lamu.data = torch.clamp(self._lamu.data, min=0., max=self.lamClamp) 
        self.lamu = self._lamu 

        self._lamr.data = torch.clamp(self._lamr.data, min=0., max=self.lamClamp)
        self.lamr = self._lamr 
        
        if self.forceAnti:
            raise NotImplementedError() 
            self.eta = -torch.exp(self._eta)
        elif self.forceHebb: 
            raise NotImplementedError() 
            self.eta = torch.exp(self._eta)
        else:
            self.eta = self._eta
            self.etau = self._etau
            self.etar = self._etar
        
        # Changes to post and pre if ignoring respective neurons for update
        if self.hebbType == 'input':
            post = torch.ones_like(post)
            postu = torch.ones_like(postu)
            postr = torch.ones_like(postr)
        elif self.hebbType == 'output':
            pre = torch.ones_like(pre)

        if self.plastic: 
            if self.groundTruthPlast: #and isFam: # only decays hebbian weights 
                raise NotImplementedError('Broke this functionality to get around something earlier.')
                A = self.lam*self.A
            elif self.updateType == 'hebb': # normal hebbian update
                Au = self.lamu*self.Au + self.etau*torch.bmm(postu, pre.unsqueeze(1)) # [B, Nh, 1] x [B, 1, Nx] = [B, Nh, Nx]
                Ar = self.lamr*self.Ar + self.etar*torch.bmm(postr, pre.unsqueeze(1))
                # (B, Nh, 1) * (Bn, Nh, Nx)
                A_new = self.lam*(postr * self.A) + self.eta*torch.bmm(post, pre.unsqueeze(1))
                A = (torch.ones_like(postu) - postu) * A_new + postu * self.A 
            elif self.updateType == 'hebb_norm': # normal hebbian update
                raise NotImplementedError() 
                # A_tilde = self.lam*self.A + self.eta*torch.bmm(post, pre.unsqueeze(1)) # Normalizes over input dimension 
                # A = A_tilde / torch.norm(A_tilde, dim=2, keepdim=True) # [B, Nh, Nx] / [B, Nh, 1]
            elif self.updateType == 'oja': # For small eta, effectively normalizes A matrix.
                raise NotImplementedError() 
                # A = self.lam*self.A + self.eta*(torch.bmm(post, pre.unsqueeze(1)) + post**2 * self.A) # [B, Nh, 1] x [B, 1, Nx] + [B, Nh, 1] * [B, Nh, Nx]
                
            if stateOnly: # This code is for functionality for finding fixed points
                return A, Au, Ar
            else:
                self.A = A
                self.Au = Au
                self.Ar = Ar
                return None, None, None

    def forward(self, x, isFam=False, debug=False, stateOnly=False):
        """
        This modifies the internal state of the model (self.A). 
        Don't call twice in a row unless you want to update self.A twice!

        x.shape: [B, Nx]
        b1.shape: [Nh]
        w1.shape=[Nx,Nh], 
        A.shape=[B,Nh,Nx], 

        """
        
        # Reset and update first, both are used to determine final h
        # b1 + (w1 + A) * x
        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)
        if self.stpType == 'add':
            a1u = torch.baddbmm(self.bu.unsqueeze(1), self.wu+self.Au, x.unsqueeze(2))
            a1r = torch.baddbmm(self.br.unsqueeze(1), self.wr+self.Ar, x.unsqueeze(2))
        elif self.stpType == 'mult':
            a1u = torch.baddbmm(self.bu.unsqueeze(1), self.wu*self.Au, x.unsqueeze(2))
            a1r = torch.baddbmm(self.br.unsqueeze(1), self.wr*self.Ar, x.unsqueeze(2))

        u = torch.sigmoid(a1u) # (B, Nh, 1)
        r = torch.sigmoid(a1r) # (B, Nh, 1)

        # print('r', r.shape)
        # print('A', self.A.shape)

        # (Nh, 1) + [(B, Nh, Nx) x (B, Nx, 1) = (B, Nh, 1)] = (B, Nh, 1) -> (B, Nh)
        if self.stpType == 'add':
            a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1+self.A, x.unsqueeze(2))
        elif self.stpType == 'mult':
            a1 = torch.baddbmm(self.b1.unsqueeze(1), self.w1*self.A, x.unsqueeze(2))

        h = self.f(a1) if self.f is not None else a1   
        # h = (torch.ones_like(u) - u) * h_new + u * self.h

        # Return is only used if finding fixed points
        A, Au, Ar = self.update_hebb(x, h, u, r, isFam=isFam, stateOnly=stateOnly)        
        if stateOnly:
            return A, Au, Ar

        if self.w2.numel()==1: # handles case where w2 has all the same elements
            w2 = self.w2.expand(1, h.shape[0])
        else:
            w2 = self.w2

        # (1, Ny) + [(B, Nh,) x (Nh, Ny) = (B, Ny)] = (B, Ny)
        a2 = self.b2.unsqueeze(0) + torch.mm(h.squeeze(dim=2), torch.transpose(self.w2, 0, 1)) #output layer activation
        y = self.fOut(a2) if self.fOut is not None else a2  
                           
        if debug: # Make a1 and h size [B, Nh]
            return a1.squeeze(dim=2), h.squeeze(dim=2), a2, y 
        return y   

#################
### Recurrent ###
#################
        
class VanillaRNN(StatefulBase):
    def __init__(self, init, verbose=True, resetState=True, netType='VanillaRNN', **rnnArgs):
        super(VanillaRNN,self).__init__()
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            W,b = random_weight_init([Nx,Nh,Nh,Ny], bias=True)

            self.n_inputs = Nx
            self.n_hidden = Nh
            self.n_outputs = Ny 
        else:
            W,b = init

        self.verbose=verbose
        init_string = 'Net parameters:\n  Net Type: {}'.format(netType)

        self.loss_fn = F.cross_entropy # Reductions is mean by default
        # self.acc_fn = binary_classifier_accuracy 
        self.acc_fn = xe_classifier_accuracy 
        
        self.fAct = rnnArgs.pop('fAct', 'sigmoid') # hidden activation
        if self.fAct  == 'linear':
            self.f = None
        elif self.fAct  == 'sigmoid':
            self.f = torch.sigmoid
        elif self.fAct  == 'tanh':
            self.f = torch.tanh
        elif self.fAct  == 'relu':
            self.f = torch.relu
        else:
            raise ValueError('f activaiton not recognized')

        self.fOutAct = rnnArgs.pop('fOutAct', 'linear') # output activiation   
        if self.fOutAct == 'linear':
            self.fOut = None # No activiation function for output
        else:
            raise ValueError('f0 activaiton not recognized')
        init_string += '\n  Structure: f: {} // fOut: {} // (Nx, Nh, Ny) = ({}, {}, {})'.format(self.fAct, self.fOutAct, Nx, Nh, Ny)

        # Determines whether or not input layer is a trainable parameter
        self.freezeInputs = rnnArgs.pop('freezeInputs', False)
        if self.freezeInputs: # Does not train input layer or hidden bias (just sets thte latter to zero)
            init_string += '\n  Winp: Frozen //  '
            self.register_buffer('w1', torch.tensor(W[0], dtype=torch.float))
            self.register_buffer('wr', torch.tensor(W[1], dtype=torch.float))
        else:
            init_string += '\n  '
            self.w1 = nn.Parameter(torch.tensor(W[0],  dtype=torch.float)) # input weights
            self.wr  = nn.Parameter(torch.tensor(W[1],  dtype=torch.float)) # recurrent weights
        self.w2  = nn.Parameter(torch.tensor(W[2],  dtype=torch.float)) # readout weights

        # Determines if hidden bias is trinable or simply not used (can overwhelm noise during delay).
        # Note this is all skipped if inputs are already frozen (since b1 is already zero)
        self.hiddenBias = rnnArgs.pop('hiddenBias', True)
        if self.hiddenBias and not self.freezeInputs:
            init_string += 'Hidden bias: trainable // '
            self.b1 = nn.Parameter(torch.tensor(b[1], dtype=torch.float)) # hidden state bias
        elif self.hiddenBias and self.freezeInputs: # Nonzero but frozen
            init_string += 'Hidden bias: frozen // '
            self.register_buffer('b1', torch.tensor(b[1], dtype=torch.float))
        else: # Frozen at zero
            init_string += 'No hidden bias // '
            self.register_buffer('b1', torch.zeros_like(torch.tensor(b[1], dtype=torch.float)))
        
        self.roBias = rnnArgs.pop('roBias', True)
        if self.roBias: # ro bias
            init_string += 'Readout bias: trainable // '
            self.b2 = nn.Parameter(torch.tensor(b[2], dtype=torch.float))
        else:
            init_string += 'No readout bias // '
            self.register_buffer('b2', torch.zeros_like(torch.tensor(b[2])))

        # Sparisfies the networks by creating masks
        self.sparsification = rnnArgs.pop('sparsification', 0.0)
        if self.sparsification > 0:
            self.register_buffer('w1Mask', torch.bernoulli((1-self.sparsification)*torch.ones_like(self.w1)))
            self.register_buffer('wrMask', torch.bernoulli((1-self.sparsification)*torch.ones_like(self.wr)))
        init_string += 'Sparsification: {:.2f} // '.format(self.sparsification)

        # Injects noise into the network
        self.noiseType = rnnArgs.pop('noiseType', None)
        self.noiseScale = rnnArgs.pop('noiseScale', 0.0)
        if self.noiseType is None:
            init_string += 'Noise type: None'
        elif self.noiseType in ('input', 'hidden',):
            init_string += 'Noise type: {} (scl: {:.1e})'.format(self.noiseType, self.noiseScale)
        else:
            raise ValueError('Noise type not recognized.')

        self.trainableState0 = rnnArgs.pop('trainableState0', False) 
        if self.trainableState0:
            # Since random_weight_init creates one extra bias we use that as the initial state
            self.h0 = nn.Parameter(torch.tensor(b[0], dtype=torch.float)) # initial h0
            init_string += '\n  h0: trainable'
        else:
            self.register_buffer('h0', torch.zeros_like(torch.tensor(b[0], dtype=torch.float)))
            init_string += '\n  h0: zeros' 

        if self.verbose:
            print(init_string)
        
        if resetState: # Don't want state to be reset when child calls it
            self.reset_state()

    def reset_state(self, batchSize=1):
        self.h = torch.ones(batchSize, *self.b1.shape, device=self.b1.device) #shape=[B,Nh]  
        # print('self:', self.device)
        # print('h:', self.h.device)
        # print('h0:', self.h0.device)
        self.h = self.h * self.h0.unsqueeze(0) # (B, Nh) x (1, Nh)
        # else:
        #     self.h = torch.zeros(batchSize, *self.b1.shape, device=self.b1.device) #shape=[B,Nh]  

    def forward(self, x, debug=False, stateOnly=False):

        

        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x+batch_noise 

        # (1, Nh) + [(B, Nx) x (Nx, Nh)] + [(B, Nh) x (Nh, Nh)] = [B, Nh]
        if self.sparsification > 0:
            a1 = (self.b1.unsqueeze(0) + 
                  torch.mm(x, torch.transpose(self.w1Mask*self.w1, 0, 1)) + 
                  torch.mm(self.h, torch.transpose(self.wrMask*self.wr, 0, 1))) 
        else:
            a1 = (self.b1.unsqueeze(0) + 
                  torch.mm(x, torch.transpose(self.w1, 0, 1)) + 
                  torch.mm(self.h, torch.transpose(self.wr, 0, 1))) 
        
        # Adds noise to the preactivation
        if self.noiseType in ('hidden',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(a1), 1/np.sqrt(self.n_hidden)*torch.ones_like(a1)
                )
            a1 = a1 + batch_noise 

        h = self.f(a1) if self.f is not None else a2
        
        if stateOnly:
            return h
        self.h = h

        # (1, Ny) + [(B, Nh) x (Nh, Ny)]
        a2 = self.b2.unsqueeze(0) + torch.mm(self.h, torch.transpose(self.w2, 0, 1))
        y = self.fOut(a2) if self.fOut is not None else a2 

        if debug:
            return a1, self.h, a2, y
        return y


    def evaluate(self, batch):
        self.reset_state(batchSize=batch[0].shape[0])
        # Output size can differ from batch[1] size because XE loss is now used 
        out_size = torch.Size([batch[1].shape[0], batch[1].shape[1], self.n_outputs]) # [B, T, Ny]
        out = torch.empty(out_size, dtype=torch.float, layout=batch[1].layout, device=batch[1].device)

        for time_idx in range(batch[0].shape[1]):

            x = batch[0][:, time_idx, :] # [B, Nx]
            out[:, time_idx] = self(x)

        return out

    @torch.no_grad()    
    def evaluate_debug(self, batch, batchMask=None, acc=True, reset=True):
        """ Returns internal parameters of the RNN """
        B = batch[0].shape[0]

        if reset:
            self.reset_state(batchSize=B)

        Nh,Nx = self.w1.shape
        Ny,Nh = self.w2.shape 

        T = batch[1].shape[1]
        db = {'x' : torch.empty(B,T,Nx),
              'a1' : torch.empty(B,T,Nh),
              'h' : torch.empty(B,T,Nh),
              'a2' : torch.empty(B,T,Ny),
              'out' : torch.empty(B,T,Ny),
              }
        for time_idx in range(batch[0].shape[1]):
            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :]

            db['x'][:,time_idx,:] = x
            db['a1'][:,time_idx], db['h'][:,time_idx,:], db['a2'][:,time_idx,:], db['out'][:,time_idx,:] = self(x, debug=True)      

        if acc:
            db['acc'] = self.accuracy(batch, out=db['out'].to(self.w1.device), outputMask=batchMask).item()  
                         
        return db

class GRU(VanillaRNN):
    def __init__(self, init, verbose=True, **rnnArgs):
        super(GRU, self).__init__(init, resetState=False, netType='GRU', verbose=verbose, **rnnArgs)
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            wz,bz = random_weight_init([Nx,Nh,Nh], bias=True)
            wr,br = random_weight_init([Nx,Nh,Nh], bias=True)     
   
        else:
            W,b = init

        self.verbose=verbose
        # Note this inherets all the parameters from the VanillaRNN, so only these new gates need to be init

        if self.freezeInputs: # All internal connections other than the readout layer are frozen
            # update gate
            self.register_buffer('wzi', torch.tensor(wz[0],  dtype=torch.float))
            self.register_buffer('wzh', torch.tensor(wz[1],  dtype=torch.float))
            self.register_buffer('bz', torch.tensor(bz[1],  dtype=torch.float))
            
            # reset gate
            self.register_buffer('wri', torch.tensor(wr[0],  dtype=torch.float))
            self.register_buffer('wrh', torch.tensor(wr[1],  dtype=torch.float))
            self.register_buffer('br', torch.tensor(br[1],  dtype=torch.float))
        else:
            # update gate
            self.wzi = nn.Parameter(torch.tensor(wz[0],  dtype=torch.float))
            self.wzh = nn.Parameter(torch.tensor(wz[1],  dtype=torch.float))
            self.bz  = nn.Parameter(torch.tensor(bz[1],  dtype=torch.float))
            
            # reset gate
            self.wri = nn.Parameter(torch.tensor(wr[0],  dtype=torch.float))
            self.wrh = nn.Parameter(torch.tensor(wr[1],  dtype=torch.float))
            self.br  = nn.Parameter(torch.tensor(br[1],  dtype=torch.float))
        
        self.reset_state()
    
    def forward(self, x, debug=False, stateOnly=False):
        
        # Add noise to the input
        if self.noiseType in ('input',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(x), 1/np.sqrt(self.n_inputs)*torch.ones_like(x)
                )
            x = x+batch_noise 

        # (1, Nh) + [(B, Nx) x (Nx, Nh)] + [(B, Nh) x (Nh, Nh)] = [B, Nh]
        # Update gate
        zgp = (self.bz.unsqueeze(0) + 
              torch.mm(x, torch.transpose(self.wzi, 0, 1)) + 
              torch.mm(self.h, torch.transpose(self.wzh, 0, 1))) 
        zg = torch.sigmoid(zgp)
        # Reset gate
        rgp = (self.br.unsqueeze(0) + 
              torch.mm(x, torch.transpose(self.wri, 0, 1)) + 
              torch.mm(self.h, torch.transpose(self.wrh, 0, 1))) 
        rg = torch.sigmoid(rgp)

        a1 = (self.b1.unsqueeze(0) + 
              torch.mm(x, torch.transpose(self.w1, 0, 1)) + 
              torch.mm(self.h * rg, torch.transpose(self.wr, 0, 1))) 
        
        # Adds noise to the preactivation
        if self.noiseType in ('hidden',):
            batch_noise = self.noiseScale*torch.normal(
                torch.zeros_like(a1), 1/np.sqrt(self.n_hidden)*torch.ones_like(a1)
                )
            a1 = a1 + batch_noise 

        h_new = self.f(a1) if self.f is not None else a2

        h = (torch.ones_like(zg) - zg) * h_new + zg * self.h
        if stateOnly:
            return h
        self.h = h

        a2 = self.b2.unsqueeze(0) + torch.mm(self.h, torch.transpose(self.w2, 0, 1))
        y = self.fOut(a2) if self.fOut is not None else a2 

        if debug:
            return zg, rg, h_new, self.h, a2, y
        return y

    @torch.no_grad()    
    def evaluate_debug(self, batch, batchMask=None):
        """ Returns internal parameters of the RNN """
        B = batch[0].shape[0]

        self.reset_state(batchSize=B)

        Nh,Nx = self.w1.shape
        Ny,Nh = self.w2.shape 

        T = batch[1].shape[1]
        db = {'x' : torch.empty(B,T,Nx),
              'zg': torch.empty(B,T,Nh),
              'rg': torch.empty(B,T,Nh),
              'h_new' : torch.empty(B,T,Nh),
              'h' : torch.empty(B,T,Nh),
              'a2' : torch.empty(B,T,Ny),
              'out' : torch.empty(B,T,Ny),
              }
        for time_idx in range(batch[0].shape[1]):
            x = batch[0][:, time_idx, :] # [B, Nx]
            y = batch[1][:, time_idx, :]

            db['x'][:,time_idx,:] = x
            db['zg'][:,time_idx], db['rg'][:,time_idx], db['h_new'][:,time_idx], db['h'][:,time_idx,:], db['a2'][:,time_idx,:], db['out'][:,time_idx,:] = self(x, debug=True)      

        db['acc'] = self.accuracy(batch, out=db['out'].to(self.w1.device), outputMask=batchMask).item()  
                         
        return db

class LSTM(VanillaRNN):
    def __init__(self, init, f=torch.tanh, fOut=torch.sigmoid):
        super(VanillaRNN,self).__init__()
        
        if all([type(x)==int for x in init]):
            Nx,Nh,Ny = init
            Wi,bi = random_weight_init([Nx,Nh,Nh], bias=True)
            Wf,bf = random_weight_init([Nx,Nh,Nh], bias=True)
            Wo,bo = random_weight_init([Nx,Nh,Nh], bias=True)
            Wc,bc = random_weight_init([Nx,Nh,Nh], bias=True) 
            Wy,by = random_weight_init([Nh,Ny], bias=True)           
        else:
            W,b = init
        
        self.Wix = nn.Parameter(torch.tensor(Wi[0],  dtype=torch.float))
        self.Wih = nn.Parameter(torch.tensor(Wi[1],  dtype=torch.float))
        self.bi  = nn.Parameter(torch.tensor(bi[1],  dtype=torch.float))
        
        self.Wfx = nn.Parameter(torch.tensor(Wf[0],  dtype=torch.float))
        self.Wfh = nn.Parameter(torch.tensor(Wf[1],  dtype=torch.float))
        self.bf  = nn.Parameter(torch.tensor(bf[1],  dtype=torch.float))
        
        self.Wox = nn.Parameter(torch.tensor(Wo[0],  dtype=torch.float))
        self.Woh = nn.Parameter(torch.tensor(Wo[1],  dtype=torch.float))
        self.bo  = nn.Parameter(torch.tensor(bo[1],  dtype=torch.float))
        
        self.Wcx = nn.Parameter(torch.tensor(Wc[0],  dtype=torch.float))
        self.Wch = nn.Parameter(torch.tensor(Wc[1],  dtype=torch.float))
        self.bc  = nn.Parameter(torch.tensor(bc[1],  dtype=torch.float))
        
        self.Wy = nn.Parameter(torch.tensor(Wy[0],  dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0],  dtype=torch.float))
        
        self.f = f
        self.fOut = fOut
        
        self.reset_state()

        
    def reset_state(self):
        self.h = torch.zeros_like(self.bc)
        self.c = torch.zeros_like(self.bc)

    
    def forward(self, x):
        #TODO: concat into single matrix for faster matmul
        ig = torch.sigmoid(torch.addmv(torch.addmv(self.bi, self.Wih, self.h), self.Wix, x)) #input gate
        fg = torch.sigmoid(torch.addmv(torch.addmv(self.bf, self.Wfh, self.h), self.Wfx, x)) #forget gate
        og = torch.sigmoid(torch.addmv(torch.addmv(self.bo, self.Woh, self.h), self.Wox, x)) #output gate
        cIn =       self.f(torch.addmv(torch.addmv(self.bc, self.Wch, self.h), self.Wcx, x)) #cell input
        self.c = fg*self.c + ig*cIn #cell state
        self.h = og*torch.tanh(self.c) #hidden layer activation i.e. cell output  
        
        y = self.fOut( torch.addmv(self.by, self.Wy, self.h) ) 
        return y
    
        
class nnLSTM(VanillaRNN): 
    """Should be identical to implementation above, but uses PyTorch internals for LSTM layer instead"""
    def __init__(self, init, f=None, fOut=torch.sigmoid): #f is ignored. Included to have same signature as VanillaRNN
        super(VanillaRNN,self).__init__()
        
        Nx,Nh,Ny = init #TODO: allow manual initialization       
        self.lstm = nn.LSTMCell(Nx,Nh)
        
        Wy,by = random_weight_init([Nh,Ny], bias=True)           
        self.Wy = nn.Parameter(torch.tensor(Wy[0],  dtype=torch.float))
        self.by = nn.Parameter(torch.tensor(by[0],  dtype=torch.float))
        
        self.loss_fn = F.binary_cross_entropy
        self.acc_fn = binary_classifier_accuracy 
        self.fOut = fOut
        
        self.reset_state()

        
    def reset_state(self):
        self.h = torch.zeros(1,self.lstm.hidden_size)
        self.c = torch.zeros(1,self.lstm.hidden_size)
    
    
    def forward(self, x):
        self.h, self.c = self.lstm(x.unsqueeze(0), (self.h, self.c))
        y = self.fOut( F.linear(self.h, self.Wy, self.by) ) 
        return y