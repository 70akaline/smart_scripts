import numpy
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf import *

def Kspace_plaquette(
    Q_IaJb_iw,
    Rs = numpy.array([[0,0],[1,0],[0,1],[1,1]]),
):   
    Ks = numpy.pi*Rs
    uKs = Rs.copy()    

    def uK_to_block(uK):
        return "(%s,%s)"%(("0" if uK[0]==0 else "pi"),("0" if uK[1]==0 else "pi")) 

    blocks = [ uK_to_block(uK) for uK in uKs]
    print blocks
    
    def get_gf_struct():
        return [[Kbl,range(2)] for Kbl in blocks]
    
    beta = Q_IaJb_iw.beta   
    npts, N_states, N_states = numpy.shape(Q_IaJb_iw.data)
    assert N_states == 8, "this is Nambu plaquette"
    assert npts % 2 == 0, "number of freqs must be divisible by 2, otherwise it's a bosonic green's function"
    npts /= 2
    nsites = N_states /2      
    
    def get_K_container():        
        gs = []
        for block in blocks: 
          gs.append ( GfImFreq(indices = range(2), beta = beta, n_points = npts, statistic = 'Fermion') )
        Q_K_ab_iw = BlockGf(name_list = blocks, block_list = gs, make_copies = True)
        return Q_K_ab_iw

    def cyclic_add(uK1,uK2):
        uK = numpy.array(uK1)+numpy.array(uK2)
        for i in range(2): 
            if uK[i]>=2: uK[i]-=2
            if uK[i]<0: uK[i]+=2
        return uK
#     for uK1 in uKs:
#         for uK2 in uKs:
#             print "%s + %s = %s"%(uK1,uK2,cyclic_add(uK1,uK2))
#     for uK1 in uKs:
#         for uK2 in uKs:
#             print "%s - %s = %s"%(uK1,uK2,cyclic_add(uK1,-uK2))

    def get_h_int(U):
        counter = 0        
        for uK in uKs:
            for uKp in uKs:    
                for uQ in uKs:
                    uK_plus_uQ = cyclic_add(uK,uQ)
                    uKp_minus_uQ = cyclic_add(uKp,-uQ)

                    uK_plus_uQ_block = uK_to_block(uK_plus_uQ)
                    uKp_minus_uQ_block = uK_to_block(uKp_minus_uQ)
                    uK_block = uK_to_block(uK)
                    uKp_block = uK_to_block(uKp)

                    h_to_add = (-U/4.0) * c_dag(uK_plus_uQ_block,0)\
                                         * c_dag(uKp_minus_uQ_block,1)\
                                         * c(uKp_block,1)\
                                         * c(uK_block,0)
                    if counter == 0:
                        h_int = h_to_add
                        counter += 1            
                    else:
                        h_int += h_to_add                                    
        return h_int

    #print get_h_int()    

    def convert_to_K_space(Q_K_ab_iw, Q_IaJb_iw):    
        for uK in uKs:        
            uK_block = uK_to_block(uK)
            for a in range(2):
                for b in range(2):
                    Q_K_ab_iw[uK_block][a,b] << 0
                    for I in range(nsites):            
                        R = Rs[I]
                        pref = numpy.exp(1j*numpy.pi*numpy.dot(uK,R))
                        #print "%s ab=%d%d I=%d R=%s pref=%s"%(uK_block,a,b,I,R,pref)
                        Q_K_ab_iw[uK_block][a,b] += pref*Q_IaJb_iw[0+a*nsites, I+b*nsites]

    def convert_to_IJ_space(Q_IaJb_iw,Q_K_ab_iw):       
        for I in range(nsites):            
            for J in range(nsites):        
                R = Rs[I]-Rs[J]
                for a in range(2):
                    for b in range(2):    
                        Q_IaJb_iw[I+a*nsites, J+b*nsites] << 0.0
                        for uK in uKs:        
                            uK_block = uK_to_block(uK)
                            pref = 0.25*numpy.exp(-1j*numpy.pi*numpy.dot(uK,R))
                            Q_IaJb_iw[I+a*nsites, J+b*nsites] += pref*Q_K_ab_iw[uK_block][a,b]
                            
    return get_K_container, get_gf_struct, get_h_int, convert_to_K_space, convert_to_IJ_space
