from modules import *
from qiskit import IBMQ
from qiskit.providers import aer

class Environment:
    
    """
    State: Wavefunction, 1 x 2^n complex vector.
    Rewards: 
        - If Steps = 100
            Return |<psi0|psif>|^2 / #steps
        - If |<psi0|psif>|^2 = 1, 
            Return 1 / #steps.
        
        Only give reward when measure
    
    """
    
    def __init__(self, num_qubits):
        
        self.provider = aer.aerprovider.AerProvider()
        self.backend = provider.get_backend('statevector_simulator')
        
        self.target_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
        
        self.reset()
            
            
    def reset(self):
        
        self.steps = 0
        self.state = np.zeros(2**num_qubits)
        self.state[0] = 1
        
        self.inner_product = np.abs(np.vdot(self.state, self.target_state))**2
            
    def step(self, action):
        
        qc = QuantumCircuit(2)
        qc.initialize(self.state, [0, 1])

        if action[0] == "X":
            qc.x(action[1])
        elif action[0] == "Y":
            qc.y(action[1])
        elif action[0] == "Z":
            qc.z(action[1])
        elif action[0] == "H":
            qc.h(action[1])
        elif action[0] == "CX":
            qc.cx(action[1][0], action[1][1])
        elif action[0] == "CCX":
            qc.ccx(action[1][0], action[1][1], action[1][2])
            
        qobj = assemble(qc)
        job = backend.run(qobj)
        result = job.result()        
        self.state = np.array(result.data()['statevector'])
        self.state = self.state[:,0] + 1j * self.state[:,1]
        
        self.get_reward()
        
        print('At end of step {}, action is {}, inner_product is {}, reward is {}'.format(self.steps, action, self.inner_product, self.reward))
        
        self.steps += 1
        
    def get_reward(self):
        
        self.inner_product = np.abs(np.vdot(self.state, self.target_state))**2
        
        if(np.abs(self.inner_product - 1) < 10e-6):
            self.reward = 1 / self.steps
        elif self.steps == 100:
            self.reward = inner_product / self.steps
        else:
            self.reward = 0
