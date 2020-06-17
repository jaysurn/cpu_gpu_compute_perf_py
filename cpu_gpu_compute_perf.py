import numpy as np
from timeit import default_timer as timer
from numba import vectorize
import logging
# parallelize code using cuda cores and push gpu_mult code to c level
@vectorize( [ 'float32( float32, float32 )' ] , target = 'cuda' )

# computations here
def gpu_mult( a , b ):
	return a ** b

def cpu_mult( a , b , c ):
	for i in range( a.size ):
		c[i] = a[i] ** b[i]
# logging to error check
def logging_m ( message , level ):
	logging.basicConfig( filename = "run_time_log.log" , format = '%(asctime)s %(message)s ' , filemode = 'w' )
	logger = logging.getLogger()

	hash_lvl = {
	"DEBUG"		: logging.DEBUG,
	"INFO"		: logging.INFO,
	"WARNING"	: logging.WARNING,
	"ERROR"		: logging.ERROR,
	"CRITICAL"	: logging.CRITICAL
	} 
	logger.setLevel ( hash_lvl [ level ] )

	logger.info( message )

	print ( message )

	return
# runs tests and times them
def run( max_vector_size ):
	curr_vector_size = 1
	run = 0
	cpu_times = []
	gpu_times = []
	while ( curr_vector_size <= max_vector_size ):
		a = b = np.array( np.random.sample( curr_vector_size ), dtype = np.float32 ) # initialize random array of nums for exponential calcs
		c = np.zeros( curr_vector_size, dtype = np.float32 ) # final output array

		start = timer()
		cpu_msg = "CPU run " + str( run ) + " with size " + str( curr_vector_size ) + ":"
		logging_m( cpu_msg , "INFO" )
		try:
			c = cpu_mult( a, b, c )
			duration = timer() - start
			cpu_times.append( round( duration, 6 ) )
			cpu_duration = duration
			cpu_stat_msg = str( duration ) + " seconds\n"
			logging_m( cpu_stat_msg , "INFO" )
		except:
			cpu_stat_msg = "Compilation failed, likely due to hitting RAM cap"
			logging_m ( cpu_stat_msg, "ERROR" )
			exit()
		g_start = timer()
		gpu_msg = "GPU run " + str( run ) + " with size " + str( curr_vector_size ) + ":"
		logging_m ( gpu_msg , "INFO" )

		try:
			c = gpu_mult( a, b )
			duration = timer() - g_start
			gpu_times.append( round( duration, 6 ) )
			gpu_duration = duration
			gpu_stat_msg = str( duration ) + " seconds\n"
			logging_m ( gpu_stat_msg , "INFO" )
		except:
			gpu_stat_msg = "Compilation failed, likely due to hitting VRAM cap"
			logging_m( gpu_stat_msg , "ERROR" )
			exit()

		update_csv( curr_vector_size , cpu_duration , gpu_duration )
		curr_vector_size *= 10
		run += 1
	return cpu_times, gpu_times
# outputs data to csv
def update_csv( vector_size , cpu_data , gpu_data ):
	msg = str( vector_size ) + "," + str( cpu_data ) + "," + str( gpu_data ) + "\n"
	log_m = msg + " has been added to the csv"

	f = open ( "runtime_output.csv" , 'a' )
	f.write( msg )
	f.close()
	logging_m( log_m , "INFO" )
	return
# prints results of tests
def print_res( cpu_res , gpu_res ):
	header = "--------------Finished--------------\nCPU Times\n" 
	logging_m( header , "INFO" )
	
	print_run = 0
	print_vec_size = 1
	for ele in cpu_res:
		cpu_out = "run " + str( print_run ) +" : " + str( ele ) + " seconds for vector size " + str( print_vec_size ) 
		logging_m( cpu_out , "INFO" )
		print_run += 1
		print_vec_size *= 10

	print ( "\nGPU times\n")
	print_run = 0
	print_vec_size = 1
	for g_ele in gpu_res:
		gpu_out = "run " + str( print_run ) + " : " + str( g_ele ) + " seconds for vector size " + str( print_vec_size )
		logging_m( gpu_out , "INFO" )
		print_run += 1
		print_vec_size *= 10
# main function, changes number of iterations by making vector_size_max smaller or bigger
def main():
	vector_size_max = 100000000
	
	output = open( "runtime_output.csv" , 'w' )
	output.write( "Vector Size,CPU,GPU\n" )
	output.close()

	cpu_times, gpu_times = run( vector_size_max )

	print_res( cpu_times, gpu_times )

main()