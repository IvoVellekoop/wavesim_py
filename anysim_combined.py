import os, time
import numpy as np
from datetime import date
from itertools import product
from scipy.linalg import eigvals
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (8,8) #(14.32,8)

class AnySim():
	def __init__(self, 
	      test='custom', 						# 'Test_1DFreeSpace', 'Test_1DGlassPlate', 'Test_2DHighContrast', 'Test_2DLowContrast', 'Test_3DHomogeneous', OR 'Test_3DDisordered'
		  n=np.ones(256), 						# Refractive index distribution
		  N_roi=np.array([256]), 				# Size of medium (in pixels)
		  lambd=1., 							# Wavelength in um (micron)
		  ppw=4, 								# points per wavelength
		  boundary_widths=np.array([20.]), 		# Width of absorbing boundaries
		  source_amplitude=1., 					# Amplitude of source
		  source_location=np.array([0]), 		# Location of source w.r.t. N_roi
		  source=None, 							# Direct source term instead of amplitude and location
		  N_domains=None,						# Number of subdomains to decompose into, in each dimension
		  overlap=np.array([20]), 				# Overlap between subdomains in each dimension
		  wrap_correction=None, 				# Wrap-around correction. None, 'L_Omega' or 'L_corr'
		  cp=20, 								# Corner points to include in case of 'L_corr' wrap-around correction
		  max_iters=int(1.1e+3)):				# Maximum number iterations

		self.test = test	# 'Test_1DFreeSpace', 'Test_1DGlassPlate', 'Test_2DHighContrast', 'Test_2DLowContrast', 'Test_3DHomogeneous', 'Test_3DDisordered'

		self.n = n
		self.N_dim = self.n.ndim				# Number of dimensions in problem

		self.N_roi = self.check_input(N_roi)	# Num of points in ROI (Region of Interest)

		self.absorbing_boundaries = True 		# True OR False
		self.boundary_widths = self.check_input(boundary_widths)

		if self.absorbing_boundaries:
			self.bw_l = np.floor(self.boundary_widths)
			self.bw_r = np.ceil(self.boundary_widths)
		else:
			self.boundary_widths = np.array([0])

		self.lambd = lambd						# Wavelength in um (micron)
		self.ppw = ppw							# points per wavelength
		self.k0 = (1.*2.*np.pi)/(self.lambd)	# wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
		self.pixel_size = self.lambd/self.ppw	# Grid pixel size in um (micron)

		self.N = self.N_roi + self.bw_l + self.bw_r

		if N_domains is None:
			self.N_domains = self.N//100	# self.N / Max permissible size of sub-domain
		else:
			self.N_domains = self.check_input(N_domains).astype(int)	# Number of subdomains to decompose into, in each dimension

		self.overlap = self.check_input(overlap).astype(int)		# Overlap between subdomains in each dimension

		if (self.N_domains == 1).all():								# If 1 domain, implies no domain decomposition
			self.domain_size = self.N.copy()
		else:														# Else, domain decomposition
			self.domain_size = (self.N+((self.N_domains-1)*self.overlap))/self.N_domains
			self.check_domain_size_max()
			self.check_domain_size_same()
			self.check_domain_size_int()

		self.bw_l = self.bw_l.astype(int)
		self.bw_r = self.bw_r.astype(int)
		self.domain_size = self.domain_size.astype(int)
		
		self.total_domains = np.prod(self.N_domains)
		self.range_total_domains = range(self.total_domains)
		
		if source is None:
			self.source_amplitude = source_amplitude
			self.source_location = source_location

			self.b = np.zeros((tuple(self.N_roi)), dtype='complex_')
			self.b[tuple(self.source_location)] = self.source_amplitude
		else:
			self.b = source

		self.crop_to_roi = tuple([slice(self.bw_l[i], -self.bw_r[i]) for i in range(self.N_dim)])

		self.wrap_correction = wrap_correction	# None, 'L_omega', OR 'L_corr'
		self.cp = cp							# number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

		self.max_iters = max_iters
		self.iters = self.max_iters - 1

		''' Create log folder / Check for existing log folder'''
		today = date.today()
		d1 = today.strftime("%Y%m%d")
		self.log_dir = 'logs/Logs_'+d1+'/'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		self.run_id = d1 + '_' + self.test
		self.run_loc = self.log_dir + self.run_id
		if not os.path.exists(self.run_loc):
			os.makedirs(self.run_loc)

		# if self.absorbing_boundaries:
		self.run_id += '_abs' + str(self.boundary_widths)
		if self.wrap_correction:
			self.run_id += '_' + self.wrap_correction
		self.run_id += '_Ndoms' + str(self.N_domains)

		self.stats_file_name = self.log_dir + self.test + '_stats.txt'

		self.print_details()	# print the simulation details
		self.init_setup()		# set up operators

	def print_details(self):
		print(f'\n{self.N_dim} dimensional problem')
		if self.test != 'custom':
			print('Test: \t\t\t', self.test)
		if self.wrap_correction:
			print('Wrap correction: \t', self.wrap_correction)
		print('Boundaries width: \t', self.boundary_widths)
		if self.total_domains > 1:
			print(f'Decomposing into {self.N_domains} domains of size {self.domain_size}, with overlap {self.overlap}')

	def init_setup(self):			# function that calls all the other 'main' functions
		# Make operators: Medium B = 1 - V, and Propagator (L+1)^(-1)
		Vraw = self.k0**2 * self.n**2
		Vraw = np.pad(Vraw, (tuple([[self.bw_l[i], self.bw_r[i]] for i in range(self.N_dim)])), mode='edge')

		self.operators = []
		if self.total_domains==1:
			self.operators.append( self.make_operators(Vraw) )
		else:
			self.operators.append( self.make_operators(Vraw[tuple([slice(0,self.domain_size[i]) for i in range(self.N_dim)])], 'left') )
			for d in range(1,self.total_domains-1):
				self.operators.append( self.make_operators(Vraw[tuple([slice(d*(self.domain_size[i]-self.overlap[i]), d*(self.domain_size[i]-self.overlap[i])+self.domain_size[i]) for i in range(self.N_dim)])], None) )
			self.operators.append( self.make_operators(Vraw[tuple([slice(-self.domain_size[i],None) for i in range(self.N_dim)])], 'right') )

		# Scale the source term (and pad if boundaries)
		if self.absorbing_boundaries:
			self.b = self.Tl * np.pad(self.b, (tuple([[self.bw_l[i], self.bw_r[i]] for i in range(self.N_dim)])), mode='constant') # source term y
		else:
			self.b = self.Tl * self.b
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.alpha = 0.75				# ~step size of the Richardson iteration \in (0,1]
		self.threshold_residual = 1.e-6
		self.iter_step = 1
	
	# Make the operators: Medium B = 1 - V and Propagator (L+1)^(-1)
	def make_operators(self, Vraw, which_end='Both'):
		N = np.squeeze(Vraw).shape
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		mu_min = (10.0/(self.boundary_widths * self.pixel_size)) if self.absorbing_boundaries else 0
		mu_min = max( np.max(mu_min), np.max(1.e+0/(np.array(N)*self.pixel_size)) )
		Vmin = np.imag( (self.k0 + 1j*np.max(mu_min))**2 )
		Vmax = 0.95
		V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 
		V0 = V0 + 1j*Vmin
		self.V = -1j*(Vraw - V0)

		# if self.wrap_correction == 'L_corr':
		# 	p = 2*np.pi*np.fft.fftfreq(N, self.pixel_size)
		# 	Lw_p = p**2
		# 	Lw = Finv @ np.diag(Lw_p.flatten()) @ F
		# 	L_corr = -np.real(Lw)
		# 	L_corr[:-self.cp,:-self.cp] = 0; L_corr[self.cp:,self.cp:] = 0  # Keep only upper and lower triangular corners of -Lw
		# 	self.V = self.V + 1j*L_corr

		self.scaling = Vmax/np.max(np.abs(self.V))
		self.V = self.scaling * self.V

		self.Tr = np.sqrt(self.scaling)
		self.Tl = 1j * self.Tr

		## B = 1 - V
		B = 1 - self.V
		if self.absorbing_boundaries and which_end != None:
			if which_end == 'Both':
				N_roi = self.N_roi
			elif which_end == 'left':
				N_roi = N - self.bw_l
			elif which_end == 'right':
				N_roi = N - self.bw_r
			else:
				N_roi = N - self.bw_l - self.bw_r
			B = self.pad_func(M=B, N_roi=N_roi, which_end=which_end).astype('complex_')
		self.medium = lambda x: B * x

		## Make Propagator (L+1)^(-1)
		if self.wrap_correction == 'L_omega':
			self.N_FastConv = N*10
		else:
			self.N_FastConv = N

		L_p = (self.coordinates_f(0, self.N_FastConv)**2)
		for d in range(1,self.N_dim):
			L_p = np.expand_dims(L_p, axis=-1) + np.expand_dims(self.coordinates_f(d, self.N_FastConv)**2, axis=0)
		L_p = 1j * self.scaling * (L_p - V0)
		Lp_inv = np.squeeze(1/(L_p+1))
		if self.wrap_correction == 'L_omega':
			self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn( np.pad(x,(0,self.N_FastConv-N)) )))[:N]
		else:
			self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn(x)))

		return [self.medium, self.propagator]

	# AnySim update
	def iterate(self):
		s1 = time.time()
		medium = 0
		propagator = 1

		## Construct restriction operators (R) and partition of unity operators (D)
		u = []
		b = []
		if self.total_domains==1:
			u.append(self.u)
			b.append(self.b)
			## To Normalize subdomain residual wrt preconditioned source
			full_normGB = np.linalg.norm( self.operators[0][medium](self.operators[0][propagator](b[0])) )
		else:
			R = []
			D = []
			ones = np.eye(self.domain_size[0])
			R_0 = np.zeros((self.domain_size[0],self.N[0]))
			for i in self.range_total_domains:
				R_mid = R_0.copy()
				R_mid[:,i*(self.domain_size[0]-self.overlap[0]):i*(self.domain_size[0]-self.overlap[0])+self.domain_size[0]] = ones
				R.append(R_mid)

			fnc_interp = lambda x: np.interp(np.arange(x), [0,x-1], [0,1])
			decay = fnc_interp(self.overlap[0])
			D1 = np.diag( np.concatenate((np.ones(self.domain_size-self.overlap), np.flip(decay))) )
			D.append(D1)
			D_mid = np.diag( np.concatenate((decay, np.ones(self.domain_size-2*self.overlap), np.flip(decay))) )
			for _ in range(1,self.total_domains-1):
				D.append(D_mid)
			D_end = np.diag( np.concatenate((decay, np.ones(self.domain_size-self.overlap))) )
			D.append(D_end)

			for j in self.range_total_domains:
				u.append(R[j]@self.u)
				b.append(R[j]@self.b)

			## To Normalize subdomain residual wrt preconditioned source
			full_normGB = np.linalg.norm(np.sum(np.array([(R[j].T @ D[j] @ self.operators[j][medium](self.operators[j][propagator](b[j]))) for j in self.range_total_domains]), axis=0))

		tj = [None for _ in self.range_total_domains]
		normb = [None for _ in self.range_total_domains]
		residual_i = [None for _ in self.range_total_domains]
		residual = [[] for _ in self.range_total_domains]
		u_iter = []
		breaker = False

		full_residual = []

		for i in range(self.max_iters):
			for j in self.range_total_domains:
				print('Iteration {}, sub-domain {}.'.format(i+1,j+1), end='\r')
				### Main update START ---
				# if i % self.iter_step == 0:
				if self.total_domains==1:
					u[j] = self.u.copy()
				else:
					u[j] = R[j] @ self.u
				tj[j] = self.operators[j][medium](u[j]) + b[j]
				tj[j] = self.operators[j][propagator](tj[j])
				tj[j] = self.operators[j][medium](u[j] - tj[j])       # subdomain residual
				### --- continued below ---

				''' Residual collection and checking '''
				## To Normalize subdomain residual wrt preconditioned source
				if self.total_domains==1:
					nr = np.linalg.norm( tj[j] )
				else:
					nr = np.linalg.norm(D[j] @ tj[j])

				residual_i[j] = nr/full_normGB
				residual[j].append(residual_i[j])

				### --- continued below ---
				u[j] = self.alpha * tj[j]
				# if i % self.iter_step == 0:
				if self.total_domains==1:
					self.u = self.u - u[j]		# instead of this, simply update on overlapping regions?
				else:
					self.u = self.u - R[j].T @ D[j] @ u[j]		# instead of this, simply update on overlapping regions?
				### Main update END ---

			### Full Residual
			if self.total_domains==1:
				full_nr = np.linalg.norm( tj[0] )
			else:
				full_nr = np.linalg.norm(np.sum(np.array([(R[j].T @ D[j] @ tj[j]) for j in self.range_total_domains]), axis=0))

			full_residual.append(full_nr/full_normGB)
			if full_residual[i] < self.threshold_residual:
				self.iters = i
				print(f'Stopping simulation at iter {self.iters+1}, residual {full_residual[i]:.2e} <= {self.threshold_residual}')
				breaker = True
				break
			self.residual_i = full_residual[i]

			if breaker:
				break
			# u_iter.append(self.u)
		self.u = self.Tr * self.u
		# self.u_iter = self.Tr.flatten() * np.array(u_iter)		## getting killed here

		# residual[1] = residual[1][::2]	# if update order is 0-1-2-1-0-... (i.e., 1 is repeated twice in one global iteration)
		self.residual = np.array(residual).T
		print(self.residual.shape)
		if self.residual.shape[0] < self.residual.shape[1]:
			self.residual = self.residual.T
		self.full_residual = np.array(full_residual)

		# Truncate u to ROI
		if self.absorbing_boundaries:
			self.u = self.u[self.crop_to_roi]
			# self.u_iter = self.u_iter[tuple((slice(None),))+self.crop_to_roi]

		self.sim_time = time.time()-s1
		print('Simulation done (Time {} s)'.format(np.round(self.sim_time,2)))

		return self.u

	def check_input(self, A):
		if not isinstance(A, np.ndarray):
			if isinstance(A, list) or isinstance(A, tuple):
				A = np.array(A)
			else:
				A = np.array([A])
		if A.shape[0] != self.N_dim:
			A = np.tile(A, self.N_dim)
		return A

	## Ensure that domain size is the same across every dim
	def check_domain_size_same(self):
		while (self.domain_size!=np.max(self.domain_size)).any():
			self.bw_r = self.bw_r + self.N_domains * (np.max(self.domain_size) - self.domain_size)
			self.N = self.N_roi + self.bw_l + self.bw_r
			self.domain_size = (self.N+((self.N_domains-1)*self.overlap))/self.N_domains

	## Ensure that domain size is less than 100 in every dim
	def check_domain_size_max(self):
		while (self.domain_size > 100).any():
			self.N_domains[np.where(self.domain_size > 100)] += 1
			self.domain_size = (self.N+((self.N_domains-1)*self.overlap))/self.N_domains

	## Ensure that domain size is int
	def check_domain_size_int(self):
		while (self.domain_size%1 != 0).any() or (self.bw_r%1 != 0).any():
			self.bw_r += np.round(self.N_domains * (np.ceil(self.domain_size) - self.domain_size),2)
			self.N = self.N_roi + self.bw_l + self.bw_r
			self.domain_size = (self.N+((self.N_domains-1)*self.overlap))/self.N_domains

	## Spectral radius
	def checkV(self, A):
		return np.linalg.norm(A,2)

	## pad boundaries
	def pad_func(self, M, N_roi, which_end='Both'):
		# boundary_ = lambda x: (np.arange(1,x+1)-0.21).T/(x+0.66)
		boundary_ = lambda x: np.interp(np.arange(x), [0,x-1], [0.04981993,0.95018007])

		for i in range(self.N_dim):
			left_boundary = boundary_(np.floor(self.bw_l[i]))
			right_boundary = boundary_(np.ceil(self.bw_r[i]))
			if which_end == 'Both':
				full_filter = np.concatenate((left_boundary, np.ones(N_roi[i]), np.flip(right_boundary)))
			elif which_end == 'left':
				full_filter = np.concatenate((left_boundary, np.ones(N_roi[i])))
			elif which_end == 'right':
				full_filter = np.concatenate((np.ones(N_roi[i]), np.flip(right_boundary)))

			M = np.moveaxis(M, i, self.N_dim-1)*full_filter
			M = np.moveaxis(M, self.N_dim-1, i)

		return M
	
	## DFT matrix
	def DFT_matrix(self, N):
		l, m = np.meshgrid(np.arange(N), np.arange(N))
		omega = np.exp( - 2 * np.pi * 1j / N )
		return np.power( omega, l * m )

	def coordinates_f(self, dimension, N):
		pixel_size_f = 2 * np.pi/(self.pixel_size*np.array(N))
		k = self.fft_range(N[dimension]) * pixel_size_f[dimension]
		return k

	def fft_range(self, N):
		return np.fft.ifftshift(self.symrange(N))

	def symrange(self, N):
		return range(-int(np.floor(N/2)),int(np.ceil(N/2)))

	## Compute and print relative error between u and some analytic/"ideal"/"expected" u_true
	def compare(self, u_true):
		# Print relative error between u and analytic solution (or Matlab result)
		self.u_true = u_true

		if self.u_true.shape[0] != self.N_roi[0]:
			self.u = self.u[tuple([slice(0,self.N_roi[i]) for i in range(self.N_dim)])]
			# self.u_iter = self.u_iter[:, :self.N_roi]

		self.rel_err = self.relative_error(self.u, self.u_true)
		print('Relative error: {:.2e}'.format(self.rel_err))
		return self.rel_err

	# Save some parameters and stats
	def save_details(self):
		print('Saving stats...')
		save_string = f'Test {self.test}; absorbing boundaries {str(self.absorbing_boundaries)}; boundaries width {self.boundary_widths}; N_domains {self.N_domains}; overlap {self.overlap}; wrap correction {self.wrap_correction}; corner points {self.cp}; {self.sim_time:>2.2f} sec; {self.iters+1} iterations; final residual {self.residual_i:>2.2e}'
		if "Test_" in self.test:
			save_string += f'; relative error {self.rel_err:>2.2e}'
		save_string += f' \n'
		with open(self.stats_file_name,'a') as fileopen:
			fileopen.write(save_string)

	# Plotting functions
	def plot_details(self):
		print('Plotting...')

		if 'Test_' in self.test:
			self.label = 'Matlab solution'
			if self.test == 'Test_1DFreeSpace':
				self.label = 'Analytic solution'

		if self.N_dim == 1:
			self.x = np.arange(self.N_roi)*self.pixel_size
			self.plot_FieldNResidual()	# png
			# if not "Test_" in self.test:
			# 	self.plot_field_iters()		# movie/animation/GIF
		elif self.N_dim == 2:
			self.image_FieldNResidual()	# png
		elif self.N_dim == 3:
			for z_slice in [0, int(self.u_true.shape[2]/2), int(self.u_true.shape[2]-1)]:
				self.image_FieldNResidual(z_slice)	# png
		plt.close('all')
		print('Plotting done.')

	def plot_basics(self, plt):
		if self.total_domains > 1:
			plt.axvline(x=(self.domain_size-self.boundary_widths-self.overlap)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
			plt.axvline(x=(self.domain_size-self.boundary_widths)*self.pixel_size, c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
			for i in range (1,self.total_domains-1):
				plt.axvline(x=((i+1)*(self.domain_size-self.overlap)-self.boundary_widths)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
				plt.axvline(x=(i*(self.domain_size-self.overlap)+self.domain_size-self.boundary_widths)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
		if "Test_" in self.test:
			plt.plot(self.x, np.real(self.u_true), 'k', lw=2., label=self.label)
		plt.ylabel('Amplitude')
		plt.xlabel('$x~[\lambda]$')
		plt.xlim([self.x[0]-self.x[1]*2,self.x[-1]+self.x[1]*2])
		plt.grid()

	def plot_FieldNResidual(self): # png
		plt.subplots(figsize=figsize, ncols=1, nrows=2)

		plt.subplot(2,1,1)
		self.plot_basics(plt)
		plt.plot(self.x, np.real(self.u), 'r', lw=1., label='AnySim')
		title = 'Field'
		if "Test_" in self.test:
			plt.plot(self.x, np.real(self.u_true-self.u)*10, 'g', lw=1., label='Error*10')
			title += f' (Relative Error = {self.rel_err:.2e})'
		plt.title(title)
		plt.legend(ncols=2, framealpha=0.6)

		plt.subplot(2,1,2)
		res_plots = plt.loglog(np.arange(1,self.iters+2, self.iter_step), self.residual, lw=1.5)
		if self.total_domains > 1:
			plt.legend(handles=iter(res_plots), labels=tuple(f'{i+1}' for i in self.range_total_domains), title='Subdomains', ncols=int(self.N_domains/4)+1, framealpha=0.5)
		plt.loglog(np.arange(1,self.iters+2, self.iter_step), self.full_residual, lw=3., c='k', ls='dashed', label='Full Residual')
		plt.axhline(y=self.threshold_residual, c='k', ls=':')
		plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
		ymin = np.minimum(6.e-7, 0.8*np.nanmin(self.residual))
		ymax = np.maximum(2.e+0, 1.2*np.nanmax(self.residual))
		plt.ylim([ymin, ymax])
		plt.title('Residual. Iterations = {:.2e}'.format(self.iters+1))
		plt.ylabel('Residual')
		plt.xlabel('Iterations')
		plt.grid()

		title_text = ''
		if self.absorbing_boundaries:
			title_text = f'{title_text} Absorbing boundaries ({self.boundary_widths}). '
		if self.wrap_correction:
			title_text = f'{title_text} Wrap correction: {self.wrap_correction}. '
		plt.suptitle(title_text)

		plt.tight_layout()
		fig_name = f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual'
		if self.wrap_correction == 'L_corr':
			fig_name += f'_cp{self.cp}'
		fig_name += f'.png'
		plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.close('all')

	def plot_field_iters(self): # movie/animation/GIF
		# self.u_iter = np.real(self.u_iter)

		fig = plt.figure(figsize=(14.32,8))
		self.plot_basics(plt)
		line, = plt.plot([] , [], 'r', lw=2., animated=True, label='AnySim')
		line.set_xdata(self.x)
		title = plt.title('')
		plt.legend()

		# Plot 100 or fewer frames. Takes much longer for any more frames.
		if self.iters > 100:
			plot_iters = 100
			iters_trunc = np.linspace(0,self.iters-1,plot_iters).astype(int)
			# u_iter_trunc = self.u_iter[iters_trunc]
		else:
			plot_iters = self.iters
			iters_trunc = np.arange(self.iters)
			# u_iter_trunc = self.u_iter

		def animate(i):
			# line.set_ydata(u_iter_trunc[i])		# update the data.
			title_text = f'Iteration {iters_trunc[i]+1}. '
			if self.absorbing_boundaries:
				title_text = f'{title_text} Absorbing boundaries ({self.boundary_widths}). '
			if self.wrap_correction:
				title_text = f'{title_text} Wrap correction: {self.wrap_correction}. '
			title.set_text(title_text)
			return line, title,
		ani = animation.FuncAnimation(
			fig, animate, interval=100, blit=True, frames=plot_iters)
		writer = animation.FFMpegWriter(
		    fps=10, metadata=dict(artist='Me'))
		ani_name = f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_Field'
		if self.wrap_correction == 'L_corr':
			ani_name += f'_cp{self.cp}'
		ani_name += f'.mp4'
		ani.save(ani_name, writer=writer)
		plt.close('all')

	def image_FieldNResidual(self, z_slice=0): # png
		if self.N_dim == 3:
			u = self.u[:,:,z_slice]
			u_true = self.u_true[:,:,z_slice]
		else:
			u = self.u.copy()
			u_true = self.u_true.copy()
		plt.subplots(figsize=figsize, ncols=2, nrows=2)
		pad = 0.03; shrink = 0.65# 1.# 

		plt.subplot(2,2,1)
		if "Test_" in self.test:
			vlim = np.maximum(np.max(np.abs(np.real(u_true))), np.max(np.abs(np.real(u_true))))
			plt.imshow(np.real(u_true), cmap='seismic', vmin=-vlim, vmax=vlim)
			plt.colorbar(shrink=shrink, pad=pad)
			plt.title(self.label)

		plt.subplot(2,2,2)
		plt.imshow(np.real(u), cmap='seismic', vmin=-vlim, vmax=vlim)
		plt.colorbar(shrink=shrink, pad=pad)
		plt.title('AnySim')

		plt.subplot(2,2,3)
		if "Test_" in self.test:
			plt.imshow(np.real(u_true-u), cmap='seismic')#, vmin=-vlim, vmax=vlim)
			plt.colorbar(shrink=shrink, pad=pad)
			plt.title(f'Difference. Relative error {self.rel_err:.2e}')

		plt.subplot(2,2,4)
		plt.loglog(np.arange(1,self.iters+2, self.iter_step), self.full_residual, lw=3., c='k', ls='dashed')
		plt.axhline(y=self.threshold_residual, c='k', ls=':')
		plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
		ymin = np.minimum(6.e-7, 0.8*np.nanmin(self.residual))
		ymax = np.maximum(2.e+0, 1.2*np.nanmax(self.residual))
		plt.ylim([ymin, ymax])
		plt.title('Residual. Iterations = {:.2e}'.format(self.iters+1))
		plt.ylabel('Residual')
		plt.xlabel('Iterations')
		plt.grid()
		plt.tight_layout()

		title_text = ''
		if self.absorbing_boundaries:
			title_text = f'{title_text} Absorbing boundaries ({self.boundary_widths}). '
		if self.wrap_correction:
			title_text = f'{title_text} Wrap correction: {self.wrap_correction}. '
		plt.suptitle(title_text)

		plt.tight_layout()
		fig_name = f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual_{z_slice}'
		if self.wrap_correction == 'L_corr':
			fig_name += f'_cp{self.cp}'
		fig_name += f'.png'
		plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.close('all')

	## Relative error
	def relative_error(self, E_, E_true):
		return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )
