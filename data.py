from pytriqs.archive import *
from pytriqs.gf import *
import numpy

class data:
  def __init__(self):
    pass

  @classmethod
  def from_file(cls, filename, dictionary):
    dt = cls()
    LoadData(dt, filename, dictionary)
    return dt

def AddScalarData(self, Qs, vals=None):
  if vals==None: vals = [0 for Q in Qs]
  for Q,val in zip(Qs,vals):
    vars(self)[Q] = val

def AddBlockScalarData(self, blocks, Qs, vals=None):
  if vals==None: vals = [ dict.fromkeys(blocks, 0) for Q in Qs]
  for Q,val in zip(Qs,vals):
    vars(self)[Q] = {}
    for block in blocks:
      vars(self)[Q][block] = val[block]

def AddNumpyData(self, Qs, shape):
  for Q in Qs:
    vars(self)[Q] = numpy.zeros(shape, dtype=numpy.complex_)

def AddBlockNumpyData(self, Qs, blocks, shape):
  for Q in Qs:
    vars(self)[Q] = {}
    for block in blocks:
      vars(self)[Q][block] = numpy.zeros(shape, dtype=numpy.complex_)

def AddGfData(self, Qs, blocks, Nsites, npts, beta, domain = 'iw', suffix='_loc_iw', statistic='Fermion'):
  gs = []
  for block in blocks: 
    if domain == 'iw': gs.append ( GfImFreq(indices = range(Nsites), beta = beta, n_points = npts, statistic = statistic) )
    if domain == 'tau': gs.append ( GfImTime(indices = range(Nsites), beta = beta, n_points = npts, statistic = statistic) )     
    if domain == 'w': assert False, "not implemented" #gs.append ( GfImFreq(indices = [0], beta = beta, n_points = niws, statistic = statistic) )     
    if domain == 't': assert False, "not implemented" #gs.append ( GfImFreq(indices = [0], beta = beta, n_points = niws, statistic = statistic) )          
  for Q in Qs:
     vars(self)[Q+suffix] = BlockGf(name_list = blocks, block_list = gs, make_copies = True)

def AddDataByConstructor(self, Qs, constructor):
  for Q in Qs:
    vars(self)[Q] = constructor()

def LoadData(self, filename, dictionary):
  try:
    A = HDFArchive(filename,'r')
  except E:
    print "data: LoadData: ERROR: cannot open archive: ",filename
    raise 
  try:
    dct = A[dictionary]
  except:
    print "data: LoadData: ERROR: dictionary not found in archive: ",dictionary
    raise 
  for Q in dct.keys():
    vars(self)[Q] = dct[Q]
  del A


def DumpData(self, filename, Qs=[], exceptions=[], dictionary='', suffix=''):
  try:
    A = HDFArchive(filename,'a')
    print "data: DumpData: opened archive",filename 
  except:
    print "data: DumpData: ERRROR: cannot open archive: ",filename
    raise 
  if dictionary!='':
     dct = {}  
     for Q in (Qs if (len(Qs)!=0) else vars(self).keys()):
       print "dumping ",Q,":",
       if isinstance(vars(self)[Q], (Gf,BlockGf)) or (not callable(vars(self)[Q])):
         if not (vars(self)[Q] is None):
           if not Q in exceptions:
             dct[Q+str(suffix)] = vars(self)[Q]
             print "...ok"        
           else: print "in exceptions!"
         else: print "is None!"
       else: print "callable and not Gf"
     A[dictionary]=dct
     #try: A[dictionary]=dct
     #except: assert False, "data: DumpData: ERRROR: cannot make dictionary: "+str(dictionary)
  else:
    for Q in (Qs if (len(Qs)!=0) else vars(self).keys()):
      try: A[Q+str(suffix)] = vars(self)[Q]
      except: print "data: DumpData: WARNING: cannot dump: ",Q 
  del A

