import os, time
import numpy as np
from datetime import date

class AnySim():
	def __init__(self, 
				n=np.ones((1,1,1)),						# Refractive index distribution
				wavelength=1., 							# Wavelength in um (micron)
				ppw=4, 									# points per wavelength
				boundary_widths=(0,0,0),	# Width of absorbing boundaries
				source=np.zeros((1,1,1)),					# Direct source term instead of amplitude and location
				N_domains=(1,1,1),						# Number of subdomains to decompose into, in each dimension
				overlap=(0,0,0),			# Overlap between subdomains in each dimension
				wrap_correction=None, 					# Wrap-around correction. None, 'L_Omega' or 'L_corr'
				cp=20, 									# Corner points to include in case of 'L_corr' wrap-around correction
				max_iters=int(1.1e+3)):						# Maximum number iterations

		AnySim.n = self.check_input_dims(n)
		AnySim.N_dims = (np.squeeze(AnySim.n)).ndim	# Number of dimensions in problem

		AnySim.N_roi = np.array(AnySim.n.shape)		# Num of points in ROI (Region of Interest)

		AnySim.boundary_widths = self.check_input_len(boundary_widths, 0)
		AnySim.bw_pre = np.floor(AnySim.boundary_widths)
		AnySim.bw_post = np.ceil(AnySim.boundary_widths)

		AnySim.wavelength = wavelength						# Wavelength in um (micron)
		AnySim.ppw = ppw							# points per wavelength
		AnySim.k0 = (1.*2.*np.pi)/(AnySim.wavelength)	# wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
		AnySim.pixel_size = AnySim.wavelength/AnySim.ppw	# Grid pixel size in um (micron)

		AnySim.N = AnySim.N_roi + AnySim.bw_pre + AnySim.bw_post
		AnySim.b = self.check_input_dims(source)

		AnySim.max_domain_size = 500
		if N_domains is None:
			AnySim.N_domains = AnySim.N//AnySim.max_domain_size				# AnySim.N / Max permissible size of sub-domain
		else:
			AnySim.N_domains = self.check_input_len(N_domains, 1)	# Number of subdomains to decompose into, in each dimension

		AnySim.overlap = self.check_input_len(overlap, 0)			# Overlap between subdomains in each dimension

		if (AnySim.N_domains == 1).all():									# If 1 domain, implies no domain decomposition
			AnySim.domain_size = AnySim.N.copy()
		else:															# Else, domain decomposition
			AnySim.domain_size = (AnySim.N+((AnySim.N_domains-1)*AnySim.overlap))/AnySim.N_domains
			self.check_domain_size_max()
			self.check_domain_size_same()
			self.check_domain_size_int()

		AnySim.bw_pre = AnySim.bw_pre.astype(int)
		AnySim.bw_post = AnySim.bw_post.astype(int)
		AnySim.N = AnySim.N.astype(int)
		AnySim.N_domains = AnySim.N_domains
		AnySim.domain_size[AnySim.N_dims:] = 0
		AnySim.domain_size = AnySim.domain_size.astype(int)

		AnySim.total_domains = np.prod(AnySim.N_domains)
		AnySim.range_total_domains = range(AnySim.total_domains)

		AnySim.crop_to_roi = tuple([slice(AnySim.bw_pre[i], -AnySim.bw_post[i]) for i in range(AnySim.N_dims)])

		AnySim.wrap_correction = wrap_correction	# None, 'L_omega', OR 'L_corr'
		AnySim.cp = cp							# number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

		AnySim.max_iters = max_iters
		AnySim.iters = AnySim.max_iters - 1

		''' Create log folder / Check for existing log folder'''
		today = date.today()
		d1 = today.strftime("%Y%m%d")
		AnySim.log_dir = 'logs/Logs_'+d1+'/'
		if not os.path.exists(AnySim.log_dir):
			os.makedirs(AnySim.log_dir)

		AnySim.run_id = d1
		AnySim.run_loc = AnySim.log_dir #+ AnySim.run_id
		if not os.path.exists(AnySim.run_loc):
			os.makedirs(AnySim.run_loc)

		AnySim.run_id += '_Ndims' + str(AnySim.N_dims) + '_abs' + str(AnySim.boundary_widths)
		if AnySim.wrap_correction:
			AnySim.run_id += '_' + AnySim.wrap_correction
		AnySim.run_id += '_Ndoms' + str(AnySim.N_domains)

		AnySim.stats_file_name = AnySim.log_dir + d1 + '_stats.txt'

		self.print_details()	# print the simulation details

	def print_details(self):
		print(f'\n{AnySim.N_dims} dimensional problem')
		if AnySim.wrap_correction:
			print('Wrap correction: \t', AnySim.wrap_correction)
		print('Boundaries width: \t', AnySim.boundary_widths)
		if AnySim.total_domains > 1:
			print(f'Decomposing into {AnySim.N_domains} domains of size {AnySim.domain_size}, with overlap {AnySim.overlap}')

	def setup_operators_n_init_variables(self):			# function that calls all the other 'main' functions
		# Make operators: Medium B = 1 - V, and Propagator (L+1)^(-1)
		Vraw = AnySim.k0**2 * AnySim.n**2
		Vraw = np.pad(Vraw, (tuple([[AnySim.bw_pre[i], AnySim.bw_post[i]] for i in range(3)])), mode='edge')

		# Vraw_blocks = [[] for _ in range(AnySim.N_dims)]

		# for j in range(AnySim.N_domains[0]):
		# 	Vraw_blocks[0].append(Vraw[j*(AnySim.domain_size[0]-AnySim.overlap):j*(AnySim.domain_size[0]-AnySim.overlap)+AnySim.domain_size[0]])

		# for i in range(AnySim.N_dims-1):
		# 	print(f'{i} main dimension')
		# 	# temp_blocks = Vraw_blocks.copy()
		# 	# Vraw_blocks = [[] for _ in range(AnySim.N_dims)]
		# 	temp_blocks = Vraw_blocks.copy()
		# 	Vraw_blocks[i] = []
		# 	print(f'range 0 to {i+1}')
		# 	for j in range(AnySim.N_dims):
		# 		print('-'*50)
		# 		print(f'{j} internal dimension for reassigning')
		# 		print(f'size of temp_blocks[{i}] = {len(temp_blocks[i])}')
		# 		for k in range(AnySim.N_domains[i]):# range(len(temp_blocks[i])):
		# 			print(f'pick element {j} in k[{i}]')
		# 			print(f'assign to Vraw_blocks[{j}], elements {k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])} to {k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])+AnySim.domain_size[i+1]}')

		# 			assign_block = temp_blocks[i][j][..., k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1]):k*(AnySim.domain_size[i+1]-AnySim.overlap[i+1])+AnySim.domain_size[i+1] ]
		# 			Vraw_blocks[j].append(assign_block)

		# 			plt.imshow(np.abs(assign_block))
		# 			plt.colorbar()
		# 			plt.show()
		# 			plt.close()

		self.operators = []
		if AnySim.total_domains==1:
			self.operators.append( self.make_operators(Vraw) )
		else:
			self.operators.append( self.make_operators(Vraw[tuple([slice(0,AnySim.domain_size[i]) for i in range(AnySim.N_dims)])], 'left') )
			for d in range(1,AnySim.total_domains-1):
				# aa = Vraw[tuple([slice(d*(AnySim.domain_size[i]-AnySim.overlap[i]), d*(AnySim.domain_size[i]-AnySim.overlap[i])+AnySim.domain_size[i]) for i in range(AnySim.N_dims)])]
				# print(d, aa.shape)
				self.operators.append( self.make_operators(Vraw[tuple([slice(d*(AnySim.domain_size[i]-AnySim.overlap[i]), d*(AnySim.domain_size[i]-AnySim.overlap[i])+AnySim.domain_size[i]) for i in range(AnySim.N_dims)])], None) )
			self.operators.append( self.make_operators(Vraw[tuple([slice(-AnySim.domain_size[i],None) for i in range(AnySim.N_dims)])], 'right') )

		# Scale the source term (and pad if boundaries)
		AnySim.b = self.Tl * np.squeeze( np.pad(AnySim.b, (tuple([[AnySim.bw_pre[i], AnySim.bw_post[i]] for i in range(3)])), mode='constant') ) # source term y
		AnySim.u = (np.zeros_like(AnySim.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		AnySim.alpha = 0.75				# ~step size of the Richardson iteration \in (0,1]
		AnySim.threshold_residual = 1.e-6
		AnySim.iter_step = 1
	
	# Make the operators: Medium B = 1 - V and Propagator (L+1)^(-1)
	def make_operators(self, Vraw, which_end='Both'):
		N = Vraw.shape
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		mu_min = (10.0/(AnySim.boundary_widths[:AnySim.N_dims] * AnySim.pixel_size)) if (AnySim.boundary_widths!=0).any() else 0
		mu_min = max( np.max(mu_min), np.max(1.e+0/(np.array(N[:AnySim.N_dims])*AnySim.pixel_size)) )
		Vmin = np.imag( (AnySim.k0 + 1j*np.max(mu_min))**2 )
		Vmax = 0.95
		V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 
		V0 = V0 + 1j*Vmin
		self.V = -1j*(Vraw - V0)

		# if AnySim.wrap_correction == 'L_corr':
		# 	p = 2*np.pi*np.fft.fftfreq(N, AnySim.pixel_size)
		# 	Lw_p = p**2
		# 	Lw = Finv @ np.diag(Lw_p.flatten()) @ F
		# 	L_corr = -np.real(Lw)
		# 	L_corr[:-AnySim.cp,:-AnySim.cp] = 0; L_corr[AnySim.cp:,AnySim.cp:] = 0  # Keep only upper and lower triangular corners of -Lw
		# 	self.V = self.V + 1j*L_corr

		self.scaling = Vmax/np.max(np.abs(self.V))
		self.V = self.scaling * self.V

		self.Tr = np.sqrt(self.scaling)
		self.Tl = 1j * self.Tr

		## B = 1 - V
		B = 1 - self.V
		if which_end == None:
			B = np.squeeze(B)
		else:
			if which_end == 'Both':
				N_roi = AnySim.N_roi
			elif which_end == 'left':
				N_roi = N - AnySim.bw_pre
			elif which_end == 'right':
				N_roi = N - AnySim.bw_post
			else:
				N_roi = N - AnySim.bw_pre - AnySim.bw_post
			B = np.squeeze(self.pad_func(M=B, N_roi=N_roi, which_end=which_end).astype('complex_'))
		self.medium = lambda x: B * x

		## Make Propagator (L+1)^(-1)
		if AnySim.wrap_correction == 'L_omega':
			self.N_FastConv = N*10
		else:
			self.N_FastConv = N

		L_p = (self.coordinates_f(0, self.N_FastConv)**2)
		for d in range(1,AnySim.N_dims):
			L_p = np.expand_dims(L_p, axis=-1) + np.expand_dims(self.coordinates_f(d, self.N_FastConv)**2, axis=0)
		L_p = 1j * self.scaling * (L_p - V0)
		Lp_inv = np.squeeze(1/(L_p+1))
		if AnySim.wrap_correction == 'L_omega':
			self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn( np.pad(x,(0,self.N_FastConv-N)) )))[:N]
		else:
			self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn(x)))

		print('B.shape', B.shape, 'Lp_inv.shape', Lp_inv.shape)

		return [self.medium, self.propagator]

	# AnySim update
	def iterate(self):
		s1 = time.time()
		medium = 0
		propagator = 1

		## Construct restriction operators (R) and partition of unity operators (D)
		u = []
		b = []
		if AnySim.total_domains==1:
			u.append(AnySim.u)
			b.append(AnySim.b)
			## To Normalize subdomain residual wrt preconditioned source
			full_normGB = np.linalg.norm( self.operators[0][medium](self.operators[0][propagator](b[0])) )
		else:
			R = []
			D = []
			ones = np.eye(AnySim.domain_size[0])
			R_0 = np.zeros((AnySim.domain_size[0],AnySim.N[0]))
			for i in AnySim.range_total_domains:
				R_mid = R_0.copy()
				R_mid[:,i*(AnySim.domain_size[0]-AnySim.overlap[0]):i*(AnySim.domain_size[0]-AnySim.overlap[0])+AnySim.domain_size[0]] = ones
				R.append(R_mid)

			fnc_interp = lambda x: np.interp(np.arange(x), [0,x-1], [0,1])
			decay = fnc_interp(AnySim.overlap[0])
			D1 = np.diag( np.concatenate((np.ones(AnySim.domain_size[0]-AnySim.overlap[0]), np.flip(decay))) )
			D.append(D1)
			D_mid = np.diag( np.concatenate((decay, np.ones(AnySim.domain_size[0]-2*AnySim.overlap[0]), np.flip(decay))) )
			for _ in range(1,AnySim.total_domains-1):
				D.append(D_mid)
			D_end = np.diag( np.concatenate((decay, np.ones(AnySim.domain_size[0]-AnySim.overlap[0]))) )
			D.append(D_end)

			for j in AnySim.range_total_domains:
				u.append(R[j]@AnySim.u)
				b.append(R[j]@AnySim.b)

			## To Normalize subdomain residual wrt preconditioned source
			full_normGB = np.linalg.norm(np.sum(np.array([(R[j].T @ D[j] @ self.operators[j][medium](self.operators[j][propagator](b[j]))) for j in AnySim.range_total_domains]), axis=0))

		tj = [None for _ in AnySim.range_total_domains]
		normb = [None for _ in AnySim.range_total_domains]
		residual_i = [None for _ in AnySim.range_total_domains]
		residual = [[] for _ in AnySim.range_total_domains]
		u_iter = []
		breaker = False

		full_residual = []

		for i in range(AnySim.max_iters):
			for j in AnySim.range_total_domains:
				print('Iteration {}, sub-domain {}.'.format(i+1,j+1), end='\r')
				### Main update START ---
				# if i % AnySim.iter_step == 0:
				if AnySim.total_domains==1:
					u[j] = AnySim.u.copy()
				else:
					u[j] = R[j] @ AnySim.u
				tj[j] = self.operators[j][medium](u[j]) + b[j]
				tj[j] = self.operators[j][propagator](tj[j])
				tj[j] = self.operators[j][medium](u[j] - tj[j])       # subdomain residual
				### --- continued below ---

				''' Residual collection and checking '''
				## To Normalize subdomain residual wrt preconditioned source
				if AnySim.total_domains==1:
					nr = np.linalg.norm( tj[j] )
				else:
					nr = np.linalg.norm(D[j] @ tj[j])

				residual_i[j] = nr/full_normGB
				residual[j].append(residual_i[j])

				### --- continued below ---
				u[j] = AnySim.alpha * tj[j]
				# if i % AnySim.iter_step == 0:
				if AnySim.total_domains==1:
					AnySim.u = AnySim.u - u[j]		# instead of this, simply update on overlapping regions?
				else:
					AnySim.u = AnySim.u - R[j].T @ D[j] @ u[j]		# instead of this, simply update on overlapping regions?
				### Main update END ---

			### Full Residual
			if AnySim.total_domains==1:
				full_nr = np.linalg.norm( tj[0] )
			else:
				full_nr = np.linalg.norm(np.sum(np.array([(R[j].T @ D[j] @ tj[j]) for j in AnySim.range_total_domains]), axis=0))

			full_residual.append(full_nr/full_normGB)
			if full_residual[i] < AnySim.threshold_residual:
				AnySim.iters = i
				print(f'Stopping simulation at iter {AnySim.iters+1}, residual {full_residual[i]:.2e} <= {AnySim.threshold_residual}')
				breaker = True
				break
			AnySim.residual_i = full_residual[i]

			if breaker:
				break
			# u_iter.append(AnySim.u)
		AnySim.u = self.Tr * AnySim.u
		# self.u_iter = self.Tr.flatten() * np.array(u_iter)		## getting killed here

		# residual[1] = residual[1][::2]	# if update order is 0-1-2-1-0-... (i.e., 1 is repeated twice in one global iteration)
		AnySim.residual = np.array(residual).T
		print(AnySim.residual.shape)
		if AnySim.residual.shape[0] < AnySim.residual.shape[1]:
			AnySim.residual = AnySim.residual.T
		AnySim.full_residual = np.array(full_residual)

		## Truncate u to ROI
		AnySim.u = AnySim.u[AnySim.crop_to_roi]
		# self.u_iter = self.u_iter[tuple((slice(None),))+AnySim.crop_to_roi]

		AnySim.sim_time = time.time()-s1
		print('Simulation done (Time {} s)'.format(np.round(AnySim.sim_time,2)))

		return AnySim.u

	def check_input_dims(self, A):
		for _ in range(3 - A.ndim):
			A = np.expand_dims(A, axis=-1)
		# if not isinstance(A, np.ndarray):
		# 	if isinstance(A, list) or isinstance(A, tuple):
		# 		A = np.array(A)
		# 	else:
		# 		A = np.array([A])
		# if A.shape[0] != AnySim.N_dims:
			# A = np.tile(A, AnySim.N_dims)
		return A
	
	def check_input_len(self, A, x):
		if isinstance(A, list) or isinstance(A, tuple):
			A += (3-len(A))*(x,)
		elif isinstance(A, int) or isinstance(A, float):
			A = tuple((A,)) + (2)*(x,)
		if isinstance(A, np.ndarray):
			A = np.concatenate((A, np.zeros(3-len(A))))
		return np.array(A).astype(int)

	## Ensure that domain size is the same across every dim
	def check_domain_size_same(self):
		while (AnySim.domain_size[:AnySim.N_dims]!=np.max(AnySim.domain_size[:AnySim.N_dims])).any():
			AnySim.bw_post[:AnySim.N_dims] = AnySim.bw_post[:AnySim.N_dims] + AnySim.N_domains[:AnySim.N_dims] * (np.max(AnySim.domain_size[:AnySim.N_dims]) - AnySim.domain_size[:AnySim.N_dims])
			AnySim.N = AnySim.N_roi + AnySim.bw_pre + AnySim.bw_post
			AnySim.domain_size[:AnySim.N_dims] = (AnySim.N+((AnySim.N_domains-1)*AnySim.overlap))/AnySim.N_domains

	## Ensure that domain size is less than 500 in every dim
	def check_domain_size_max(self):
		while (AnySim.domain_size > AnySim.max_domain_size).any():
			AnySim.N_domains[np.where(AnySim.domain_size > AnySim.max_domain_size)] += 1
			AnySim.domain_size = (AnySim.N+((AnySim.N_domains-1)*AnySim.overlap))/AnySim.N_domains

	## Ensure that domain size is int
	def check_domain_size_int(self):
		while (AnySim.domain_size%1 != 0).any() or (AnySim.bw_post%1 != 0).any():
			AnySim.bw_post += np.round(AnySim.N_domains * (np.ceil(AnySim.domain_size) - AnySim.domain_size),2)
			AnySim.N = AnySim.N_roi + AnySim.bw_pre + AnySim.bw_post
			AnySim.domain_size = (AnySim.N+((AnySim.N_domains-1)*AnySim.overlap))/AnySim.N_domains

	## Spectral radius
	def checkV(self, A):
		return np.linalg.norm(A,2)

	## pad boundaries
	def pad_func(self, M, N_roi, which_end='Both'):
		# boundary_ = lambda x: (np.arange(1,x+1)-0.21).T/(x+0.66)
		boundary_ = lambda x: np.interp(np.arange(x), [0,x-1], [0.04981993,0.95018007])

		for i in range(AnySim.N_dims):
			left_boundary = boundary_(np.floor(AnySim.bw_pre[i]))
			right_boundary = boundary_(np.ceil(AnySim.bw_post[i]))
			if which_end == 'Both':
				full_filter = np.concatenate((left_boundary, np.ones(N_roi[i]), np.flip(right_boundary)))
			elif which_end == 'left':
				full_filter = np.concatenate((left_boundary, np.ones(N_roi[i])))
			elif which_end == 'right':
				full_filter = np.concatenate((np.ones(N_roi[i]), np.flip(right_boundary)))

			M = np.moveaxis(M, i, -1)*full_filter
			M = np.moveaxis(M, -1, i)

		return M
	
	## DFT matrix
	def DFT_matrix(self, N):
		l, m = np.meshgrid(np.arange(N), np.arange(N))
		omega = np.exp( - 2 * np.pi * 1j / N )
		return np.power( omega, l * m )

	def coordinates_f(self, dimension, N):
		pixel_size_f = 2 * np.pi/(AnySim.pixel_size*np.array(N))
		k = self.fft_range(N[dimension]) * pixel_size_f[dimension]
		return k

	def fft_range(self, N):
		return np.fft.ifftshift(self.symrange(N))

	def symrange(self, N):
		return range(-int(np.floor(N/2)),int(np.ceil(N/2)))

	## Compute and print relative error between u and some analytic/"ideal"/"expected" u_true
	def compare(self, u_true):
		# Print relative error between u and analytic solution (or Matlab result)
		AnySim.u_true = u_true

		if AnySim.u_true.shape[0] != AnySim.N_roi[0]:
			AnySim.u = AnySim.u[tuple([slice(0,AnySim.N_roi[i]) for i in range(AnySim.N_dims)])]
			# self.u_iter = self.u_iter[:, :AnySim.N_roi]

		AnySim.rel_err = self.relative_error(AnySim.u, AnySim.u_true)
		print('Relative error: {:.2e}'.format(AnySim.rel_err))
		return AnySim.rel_err

	## Relative error
	def relative_error(self, E_, E_true):
		return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )