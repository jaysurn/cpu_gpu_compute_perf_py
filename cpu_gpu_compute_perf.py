import numpy as np
from timeit import default_timer as timer
from numba import vectorize

# parallelize code using cuda cores and push gpu_mult code to c level
@vectorize( [ 'float32( float32, float32 )' ], target = 'cuda' )

# computations here
def gpu_mult( a, b ):
	return a ** b

def cpu_mult( a, b ,c ):
	for i in range( a.size ):
		c[i] = a[i] ** b[i]
# runs tests and times them
def run( max_vector_size ):
	curr_vector_size = 1
	run = 0
	cpu_times = []
	gpu_times = []
	while ( curr_vector_size <= max_vector_size ):
		a = b = np.array( np.random.sample(curr_vector_size), dtype = np.float32 )
		c = np.zeros( curr_vector_size, dtype = np.float32 )

		start = timer()
		print ( "CPU run " + str( run ) + " with size " + str( curr_vector_size ) + ":" )
		c = cpu_mult( a, b, c )
		duration = timer() - start
		cpu_times.append( round( duration, 6 ) )
		print ( str( duration ) + " seconds\n" )

		g_start = timer()
		print ( "GPU run " + str( run ) + " with size " + str( curr_vector_size ) + ":" )
		c = gpu_mult( a, b )
		duration = timer() - g_start
		gpu_times.append( round( duration, 6 ) )
		print ( str( duration ) + " seconds\n" )
		curr_vector_size *= 10
		run += 1

	return cpu_times, gpu_times
# prints results of tests
def print_res( cpu_res, gpu_res ):
	print ( "--------------Finished--------------\nCPU Times\n" )
	
	print_run = 0
	print_vec_size = 1
	for ele in cpu_res:
		print ( "run " + str( print_run ) +" : " + str( ele ) + " seconds for vector size " + str( print_vec_size ) )
		print_run += 1
		print_vec_size *= 10

	print ( "\nGPU times\n")
	print_run = 0
	print_vec_size = 1
	for g_ele in gpu_res:
		print ( "run " + str( print_run ) + " : " + str( g_ele ) + " seconds for vector size " + str( print_vec_size ) )
		print_run += 1
		print_vec_size *= 10
# main function, changes number of iterations by making vector_size_max smaller or bigger
def main():
	vector_size_max = 100000000
	
	cpu_times, gpu_times = run( vector_size_max )

	print_res( cpu_times, gpu_times )

main()