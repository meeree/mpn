# Based off code from: https://github.com/google-research/reverse-engineering-neural-networks/blob/master/renn/rnn/fixed_points.py

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import copy

from scipy.spatial import distance


class fixedPointNetwork(nn.Module): 
    """
    Creates a network in order to find the fixed points of the internal state

    This takes advantage of PyTorch's ability to find derivatives of any network implemented in 
    its langauge. In this "network", the parameters of the network being analyzed are frozen and
    the parameters of this network are the states for which we wish to find fixed points. We then
    train this network with MSE loss, where the "labels" are the previous states
    """


    def __init__(self, network, init_states):
        """Subclasses must either implement self.loss_fn or override self.average_loss()"""
        super(fixedPointNetwork, self).__init__()
        self.eval() #start the module in evaluation mode
        
        self.name = self.__class__.__name__

        # Creates a copy of the network and freezes its parameters
        net_fzn = copy.deepcopy(network)
        for param in net_fzn.parameters():
            param.requires_grad = False
        self.net = net_fzn
        self.netType = net_fzn.name

        # Parameters are states
        self.states = nn.Parameter(torch.tensor(init_states, dtype=torch.float))

        print('FP Network - NetType: {}, Points size: {}'.format(
            self.netType, self.states.shape)
        )

    def forward(self, x, currentStates=None):
        """ 
        Given an input and the current pts, returns the next state
        using the underlying network 

        Note the internal state of the network is set to the states being optimized by default,
        but this can also be used to calculate a single-forward pass update of any given current
        state by passing them into currentStates (this is used for comparing speeds for example)

        """

        if self.netType in ('VanillaRNN', 'GRU',):  
            self.net.h = self.states if currentStates is None else currentStates
        elif self.netType == 'HebbNet':
            self.net.A = self.states if currentStates is None else currentStates
        else:
            raise ValueError('NetType not recoginzed')

        return self.net(x, stateOnly=True)

    def get_speeds(self, inputs, currentStates=None):
        with torch.no_grad():
            next_state = self(inputs, currentStates=currentStates)
            if self.netType in ('VanillaRNN', 'GRU',): 
                norm_dims = (1,)
            elif self.netType == 'HebbNet':
                norm_dims = (1,2,)

            if currentStates is None:
                return 1/2 * (torch.norm(next_state - self.states, dim=norm_dims)**2).cpu().numpy() #[B,]
            else:
                return 1/2 * (torch.norm(next_state - currentStates, dim=norm_dims)**2).cpu().numpy() #[B,]
    

    def find_fixed_points(self, inputs, steps, **kwargs):

        init_speeds = self.get_speeds(inputs)
        print('Init speeds - Max: {:.2e} /Min: {:.2e}'.format(max(init_speeds), min(init_speeds)))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs.pop('learningRate', 1e-3))
        printPeriod = kwargs.pop('printPeriod', 10)    

        # Store loss history.
        loss_hist = []
        for step in range(steps):

            self.optimizer.zero_grad()

            next_state = self(inputs) # expects shape [B, state_size], out: [B, state_size]

            # Labels of loss are the input states
            loss = F.mse_loss(next_state, self.states, reduction='mean')
            loss_hist.append(loss.item())

            loss.backward()

            self.optimizer.step()

            if step % printPeriod == 0:
                print('  Step {} - Loss: {:.3e}'.format(step, loss.item())) 

        final_speeds = self.get_speeds(inputs)
        print('Final speeds - Max: {:.2e} /Min: {:.2e}'.format(max(final_speeds), min(final_speeds)))

        return self.states.detach(), loss_hist, final_speeds




#### OLD CODE FROM JAX SETUP ######


def build_fixed_point_loss(net):
    """Builds function to compute speed of hidden states.

    Args:
      net: a stateful network of some kind
      net_params: network parameters to use when applying the network.

    Returns:
      fixed_point_loss_fun: function that takes a batch of hidden states
        and inputs and computes the speed of the corresponding hidden
        states.
    """

    # Creates a copy of the network and freezes its parameters in place
    net_fzn = copy.deepcopy(net)

    for param in net_fzn.parameters():
        param.requires_grad = False 

    def fixed_point_loss_fun(state, x):
        """Computes the speed of hidden states.

        The speed is defined as the squared l2 distance between
        the current state and the next state, in response to a given
        input:

          Q = (1/2) || h - F(h, x) ||_2^2

        Args:
          state: The current state of the network (differs for each network)
          x: The current input as a vector.

        Returns:
          fixed_point_loss_fun: A function that computes the fixed point speeds
            for a list or array of states.
        """
        
        # Sets the current state of the network to compute next hidden
        if net_params['netType'] in ('HebbNet',):
            h, A = state # State of network determined by A instead of h
            net.A = A
        elif net_params['netType'] in ('VanillaRNN',):
            h = state
            net.h = h

        # Single forward pass to get next hidden
        db = net(x, debug=True)

        # Hiddens are always [B, Nh]
        return 0.5 * torch.sum((h - db['h'])**2, -1)

    return fixed_point_loss_fun


def find_fixed_points(fp_loss_fun,
                      initial_states,
                      x_star,
                      tolerance,
                      steps=range(1000),
                      lr=1e-3):
  """Run fixed point optimization.

  Args:
    fp_loss_fun: Function that computes fixed point speeds.
    initial_states: Initial state seeds.
    x_star: Input at which to compute fixed points.
    optimizer: A jax.experimental.optimizers tuple.
    tolerance: Stopping tolerance threshold.
    steps: Iterator over steps.

  Returns:
    fixed_points: Array of fixed points for each tolerance.
    loss_hist: Array containing fixed point loss curve.
    squared_speeds: Array containing the squared speed of each fixed point.
  """
  loss_hist, fps = optimize_loss(lambda s: torch.mean(fp_loss_fun(s, x_star)),
                                 initial_states,
                                 steps,
                                 lr=lr,
                                 stop_tol=tolerance)

  fixed_points = fps
  squared_speeds = fp_loss_fun(fps, x_star)

  return fixed_points, loss_hist, squared_speeds


def exclude_outliers(points, threshold=np.inf, verbose=False):
  """Remove points that are not within some threshold of another point."""

  # Return all fixed points if tolerance is <= 0
  if np.isinf(threshold):
    return points

  # Return if there are less than two fixed points.
  if points.shape[0] <= 1:
    return points

  # Compute pairwise distances between all fixed points.
  distances = distance.squareform(distance.pdist(points))

  # Find distance to each nearest neighbor.
  nn_distance = np.partition(distances, 1, axis=0)[1]

  # Keep points whose nearest neighbor is within some distance threshold.
  keep_indices = np.where(nn_distance <= threshold)[0]

  # Log how many points were kept.
  if verbose:
    print(f'Keeping {len(keep_indices)} out of {len(points)} points.')

  return points[keep_indices]

def optimize_loss(loss_fun, s0, steps, lr=1e-3, stop_tol=-np.inf):
  """Run an optimizer on a given loss function.
  Args:
    loss_fun: Scalar loss function to optimize.
    s0: Initial states.
    steps: Iterator over steps.
    stop_tol: Stop if the loss is below this value (Default: -np.inf).
  Returns:
    loss_hist: Array of losses during training.
    final_params: Optimized parameters.
  """

  # Initialize optimizer.
  optimizer = torch.optim.Adam(self.parameters(), lr=lr) 

  loss_hist = []
  for k in steps:
    optimizer.zero_grad()
    loss = loss_fun(s0)

    loss.backward()
    optimizer.step() 

  opt_state = init_opt(x0)

  # -------- JAX Version --------

  # Loss and gradient.
  value_and_grad = jax.value_and_grad(loss_fun)

  @jax.jit
  def step(k, state):
    params = get_params(state)
    loss, grads = value_and_grad(params)
    return loss, update_opt(k, grads, state)

  # Store loss history.
  loss_hist = []
  for k in steps:
    f, opt_state = step(k, opt_state)
    loss_hist.append(f)

    if f <= stop_tol:
      break

  # Extract final parameters.
  final_params = get_params(opt_state)

  return np.array(loss_hist), final_params