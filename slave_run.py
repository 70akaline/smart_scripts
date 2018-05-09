import pytriqs.utility.mpi as mpi

def slave_run(data_package, printout=True, tasks = {}):
  while True:
    if printout: print "[Node ",mpi.rank,"] waiting for instructions..."

    data_package = mpi.bcast(data_package)

    if printout: print "[Node ",mpi.rank,"] received instructions!!!"

    if data_package is None: 
      if printout: print "[Node ",mpi.rank,"] data_package is None, will exit now. Goodbye."          
      break

    if data_package['tag'] in tasks.keys(): 
      tasks[data_package['tag']](data_package)
    elif data_package['tag']=='exit': 
      break
    else:
      print "[Node ",mpi.rank,"] ERROR: unknown task tag!!!!" 
