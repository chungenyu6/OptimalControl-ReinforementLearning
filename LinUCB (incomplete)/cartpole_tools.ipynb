import numpy as np
import gym
import numpy.random as rnd
from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import logger
import scipy.linalg as la

class SwingUp(CartPoleEnv):
    """
    This is a modification of the gym environment where 
    - the action space is all real numbers (real vectors of length 1)
    - the pendulum starts near the down position

    The goal is to use this for testing swing-up algorithms
    """
    def __init__(self):
        super().__init__()
        
        action_high = np.array([np.finfo(np.float32).max])
        self.action_space = gym.spaces.Box(-action_high,action_high,dtype = np.float32)
        self.sigma_x = .01
        self.sigma_omega = .01
    def step(self, action):
        """
        The equations in the openai gym cartpole environment do not match those of Lozano and Fatoni.
        The Lozano and Fatoni equations are derived directly from Lagrange's equations,
        and I trust them more than force balance approach used in OpenAI gym, which comes from
        
        https://coneural.org/florian/papers/05_cart_pole.pdf
        
        So, we will use the equations of motion from Lozano and Fatoni.
        """
        
        
        x, x_dot, theta, theta_dot = self.state
        
        
        
        #force = self.force_mag if action == 1 else -self.force_mag
        
        force = action[0]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        MassMatrix = np.array([[self.total_mass,self.polemass_length * costheta],
                               [self.polemass_length * costheta, self.polemass_length * self.length]])
        
        CoriolisMatrix = np.array([[0,-self.polemass_length * theta_dot * sintheta],
                                   [0,0]])
        
        GravityVector = np.array([0,
                                 -self.polemass_length * self.gravity * sintheta])
        
        tau = np.array([force,0])
        q_dot = np.array([x_dot,
                          theta_dot])
        q_ddot = la.solve(MassMatrix,tau - CoriolisMatrix@q_dot - GravityVector)
        
        xacc,thetaacc = q_ddot

        tauRt = np.sqrt(self.tau)
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc + tauRt * self.sigma_x * rnd.randn()
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc + tauRt * self.sigma_omega * rnd.randn()
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

     
        done = False
            
        reward = 0

        return np.array(self.state, dtype=np.float32), reward, done, {}
    
    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) + \
        np.array([0,0,np.pi,0])
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

class SwingUpController:
    def __init__(self,k_x=1,k_v=1.,Q = np.eye(4),R = 1e0 * np.eye(1)):
        
        self.k_x = k_x
        self.k_v = k_v
        self.Q = Q
        self.R = R
        self.update_parameters(0,0,0,0,0)
        
        self.k_p_baseline = 1.
        self.k_d_baseline = 10.
        self.x_bound = 2.
        self.theta_dot_bound = 10.
        self.fallen = True
        self.fall_bound = .2
        
    def update_parameters(self,h,m,M,g,ell):
        self.m = m
        self.h = h
        self.M = M
        self.g = g
        self.ell = ell
        
        self.param = np.array([h,m,M,g,ell])
        
    def MassMatrix(self,theta):
        h,m,M,g,ell = self.param
        MassMat = np.array([[m+M,m*ell*np.cos(theta)],
                            [m*ell*np.cos(theta),m*ell**2]])
        
        return MassMat    
        
        
    def action(self,s):
        
        x,x_dot,theta,theta_dot = s
     
            
        h,m,M,g,ell = self.param
                
        
        MassMat = self.MassMatrix(theta)
                
        q_dot = np.array([x_dot,theta_dot])
        E = .5 * q_dot@MassMat@q_dot + m*g*ell*(np.cos(theta)-1)
        
        EBound = np.min([2*m*g*ell,.9 * self.k_v/2])
        
        
        use_baseline = (np.abs(x) >  self.x_bound) or \
        (np.abs(E) > EBound )
        
        if use_baseline:
            # This is basic PD controller to keep the 
            # system within a reasonable range
            a = np.array([-self.k_p_baseline * x - self.k_d_baseline * x_dot ])
            #print('reset')
        else:
            if np.min(self.param) <= 0:
                # This is a pure exploration controller used when the parameter 
                # estimates are bad.
                a = 10 * rnd.randn(1)
                #print('random!')
            else:
                   
                if np.abs(np.sin(theta)) > self.fall_bound:
                    self.fallen = True
                    
                if self.fallen and np.abs(np.sin(theta)) <= self.fall_bound:
                    # Pendulum has swung up
                    # Calculate feedback a new feedback and try to stabilize
                    self.fallen = False
                    
                    MassMat0 = self.MassMatrix(0.)
                    G_mat = np.array([[0,0,0,0],
                                      [0,-m*g*ell,0,0]])
                    
                    # Build the A and B matrices in more standard coordinates for
                    # Lagrange's equations
                    A_tilde_top = np.hstack([np.zeros((2,2)),np.eye(2)])
                    A_tilde_bot = la.solve(MassMat0,-G_mat)
                    A_tilde = np.vstack([A_tilde_top,
                                         A_tilde_bot])
                    
                    Force_mat = np.array([[1.],
                                          [0.]])
                    B_tilde = np.vstack([ np.zeros((2,1)), la.solve(MassMat0,Force_mat)])
                    
                    # Permutation matrix to bring the system into coordinates used by gym environment
                    P = np.array([[1.,0,0,0],
                                  [0,0,1.,0],
                                  [0,1.,0,0],
                                  [0,0,0,1.]])
                    
                    A = np.eye(4) + h * P@A_tilde @ P.T 
                    B = h * P @ B_tilde
                    
                    X = la.solve_discrete_are(A,B,self.Q,self.R)
                    self.K = -la.solve(self.R + B.T@X@B,B.T@X@A)
                
                
                if self.fallen:
                    SinDen = M+m*np.sin(theta)**2
                
                    a_den = E + self.k_v / SinDen
                    a_num = -x_dot - self.k_x * x - \
                    self.k_v * m * np.sin(theta) * (ell * theta_dot**2 - g*np.cos(theta)) / SinDen
                
                    a = np.array([ a_num / a_den ])
                    #print('swing up')
                    
                else:
                    # Translate the angle to be in [-pi,pi]
                    s_trans = np.copy(s)
                    s_trans[2] = ((theta + np.pi) % (2*np.pi)) - np.pi
                    a = self.K @ s_trans
                    #print('stabilize')
                    
        return a
        
