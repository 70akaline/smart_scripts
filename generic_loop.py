import numpy
from numpy import matrix, array, zeros
from pytriqs.operators import *
from pytriqs.archive import *
from pytriqs.gf.local import *
from pytriqs.arrays import BlockMatrix, BlockMatrixComplex
import pytriqs.utility.mpi as mpi
from time import *

def print_timings(name, times):    
  for l in range(len(times)-1):
    print times[l][1], " took: ", times[l+1][0]-times[l][0]," secs"
  print "whole ",name," took: ", times[-1][0]-times[0][0], " secs"

def short_timings(times):    
  s = ""
  for l in range(len(times)-1):
    dt = times[l+1][0]-times[l][0]
    if dt>1:
      s+= "%s %.2f secs; "%(times[l][1], dt)
  return s
#------------------------------------------- GERNERIC ACTION ---------------------------------------------------#

class generic_action:
  def __init__( self,
                name = "generic action",
                main = lambda data: 0,
                mixers = [lambda data, it: 0],
                cautionaries = [lambda data, it: 0], allowed_errors = [],               
                printout = lambda data, it: 0,
                short_timings = True):
                 
    self.name = name
    self.main = main
    self.mixers = mixers
    for mixer in mixers:
      mixer.get_initial()
    self.cautionaries = cautionaries
    self.allowed_errors = allowed_errors
    self.printout = printout
    self.err = False
    self.errs = [False for c in cautionaries] 

  def execute(self, data, it):
    if mpi.is_master_node(): print "-------------------------", self.name

    try:     
      times = [(time(),"main")]          
      self.main(data)

      times.append((time(),"mixing"))
      for mixer in self.mixers:
        mixer.mix(it)

      times.append((time(),"cautionaries"))
      self.err = False
      for caut in self.cautionaries:
        ind = self.cautionaries.index(caut)  
        err = self.errs[ind] = caut(data, it)
        self.err = ( (err and (not (ind in self.allowed_errors))) or self.err) #if already true, do not revert back to false        

      times.append((time(),"printout"))
      self.printout(data,'it-%.2d'%it)

      times.append((time(),""))
      if mpi.is_master_node():
         s=""
         if not short_timings:
           print "###########", self.name, " timings"
           print_timings(self.name, times) 
           print "###########"
         else: s=short_timings(times) 
         print "-------------------------",self.name,"done!",s,("!!!!>>> there were errors..." if self.err else "")
    except Exception as e:
      print e
      print "[ Node",mpi.rank,"] ERROR: action",self.name,"broke"
      import traceback, os.path, sys
      top = traceback.extract_stack()[-1]
      if mpi.is_master_node():
        data.dump('broken')
      raise  

    return self.err


#------------------------------------------- GERNERIC LOOP ---------------------------------------------------#
class generic_loop:
  def __init__( self,
                name = "generic loop", 
                actions = [],
                convergers = [],  
                monitors = []):
    self.name = name 
    self.actions = actions
    self.convergers = convergers
    self.monitors = monitors

  def run(self, data, 
                max_its = 10,
                min_its = 5, 
                max_it_err_is_allowed = 7,
                print_final = True,
                print_current = 5,
                start_from_action_index = 0 ):
    if mpi.is_master_node(): print "============================== running ",self.name, "================================="

    for conv in self.convergers:
      conv.reset()
    for monitor in self.monitors:
      monitor.reset()

    for it in range(max_its):
      if mpi.is_master_node(): print "------------------- iteration ",it,"/",max_its,"---------------------"
      times = []
       
      err = False       
      for action in (self.actions if it>0 else self.actions[start_from_action_index:]):
        times.append((time(),action.name))
        err = (action.execute(data, it) or err)

      times.append((time(),"convergers"))
      converged = True   
      for conv in self.convergers:
        converged = (conv.check() and converged) #all must converge

      times.append((time(),"monitors"))
      for monitor in self.monitors:
        monitor.monitor()

      times.append((time(),""))
      if mpi.is_master_node():
        header_lbl = "################## "+ self.name+ " timings ################"
        footer_lbl = ""
        for l in range(len(header_lbl)):
          footer_lbl += "#"
        print header_lbl
        print_timings("iteration", times) 
        print footer_lbl
   
      if converged and it>=min_its:
        if mpi.is_master_node(): print "=-=-=-=-=-=-=-=-=-=-=-=-", self.name, " converged!!"
        break 
      if err and (it>max_it_err_is_allowed):
        if mpi.is_master_node(): print "=-=-=-=-=-=-=-=-=-=-=-=-", self.name, " erroneous!!! exiting..."
        break

      if mpi.is_master_node():
        if (it + 1) % print_current == 0: data.dump('current')
    if mpi.is_master_node():
      if print_final: data.dump('final')      

    return err

################################# CONVERGENCE and MIXING ###############################################
import copy
class mixer:
  def __init__(self, mixed_quantity, rules=[[0, 0.0], [5, 0.3], [15, 0.65]], func=None, initialize = False):
    self.rules = rules #rules are expected to be in ascending order of the starting interation which is the first element in the sublists (rule: [starting iteration, ratio])
    self.mq = mixed_quantity #for now only a single bosonic matrix valued BlockGf can be monitored for convergence and mixed
    self.func = func
    if initialize: self.get_initial()

  def get_initial(self):
    self.mq_old = copy.deepcopy(self.mq())

  def mix(self, loop_index):
    #mix the monitored bosonic Gf
    ratio = 0.0
    for rule in self.rules:
      if loop_index>=rule[0]:
        ratio = rule[1]

    if self.func is None:
      self.mix_gf(ratio)
    else:
      self.func(self, ratio) 

    del self.mq_old
    self.get_initial()

  def mix_block_gf(self, ratio):
    if mpi.is_master_node(): print "mixer: ratio:",ratio
    for name, m in self.mq():
      m << ratio*self.mq_old[name] + (1.0-ratio)*m

  def mix_gf(self, ratio):
    self.mq().data[:,0,0] = ratio*self.mq_old.data[:,0,0] + (1.0-ratio)*self.mq().data[:,0,0]


  #def mix_regular(self, ratio): #THIS IS NOT GOING TO WORK #for now works only with mutable objects
  #  self.mq = ratio*self.mq_old + (1.0-ratio)*self.mq()

  def mix_lattice_gf(self, ratio):
    if mpi.is_master_node(): print "mixer: ratio:",ratio
    for key in self.mq().keys():
      self.mq()[key][:,:,:] = ratio*self.mq_old[key][:,:,:] + (1.0-ratio)*self.mq()[key][:,:,:]

  #def mix_dictionary(self, ratio):
  #  for key in self.mq().keys():
  #    self.mq()[key] = ratio*self.mq_old[key] + (1.0-ratio)*self.mq()[key


class converger:
  def __init__(self, monitored_quantity, accuracy=3e-5, func=None, archive_name=None, h5key='diffs'):
    #monitored quantity needs to be a function returning an object in case the object is rewritten (changed address)
    self.mq = monitored_quantity

    self.accuracy = accuracy
    self.diffs = []
    self.func = func

    self.archive_name = archive_name
    self.h5key = h5key

    if mpi.is_master_node(): print "converger initiialized: archive_name: %s h5key: %s accr: %s"%(archive_name,h5key,accuracy) 

  def reset(self):
    self.get_initial()
    self.diffs = []

  def get_initial(self):
    self.mq_old = copy.deepcopy(self.mq())

  def check(self):
    if self.func is None:
      self.check_gf()
    else:
      self.func(self)

    del self.mq_old
    self.get_initial()
    
    if mpi.is_master_node(): 
      print "converger: ",self.h5key," : ", self.diffs[-1]
      if (not (self.archive_name is None)):
        A = HDFArchive(self.archive_name)
        A[self.h5key] = self.diffs
        del A

    if self.diffs[-1]<self.accuracy:
      return True

    return False

  def check_gf(self):
    max_diff = 0 
    mq = self.mq() 
    for key, m in mq: 
      diff = abs(m.data[:,:,:] - self.mq_old[key].data[:,:,:])                
      md = numpy.amax(diff)
      if md > max_diff: max_diff = md
    self.diffs.append(max_diff)         

  def check_numpy_array(self):
    max_diff = 0 
    diff = numpy.abs(self.mq() - self.mq_old)                
    md = numpy.amax(diff)
    if md > max_diff: max_diff = md
    self.diffs.append(max_diff)         


class monitor:
  def __init__(self, monitored_quantity, h5key, func=None, struct=None, archive_name=None):
    #monitored quantity needs to be a function returning an object in case the object is rewritten (changed address)
    self.mq = monitored_quantity
    self.h5key = h5key

    self.func = func

    self.archive_name = archive_name
    self.struct = struct

  def reset(self):
    self.values = []

  def monitor(self):
    try:  
      self.values.append(copy.deepcopy(self.mq()))    
    except Exception as e:
      if mpi.is_master_node():
        print "monitor: ",self.h5key," cound not read the value. appending nan..."
        print e
      self.values.append(float('nan'))    
    if mpi.is_master_node() and (not (self.archive_name is None)):
      A = HDFArchive(self.archive_name)
      A[self.h5key] = self.values
      del A
      print "monitor: ",self.h5key," : ", self.values[-1]

     
