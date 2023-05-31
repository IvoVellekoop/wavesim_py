import os, time
import numpy as np
from datetime import date
from scipy.linalg import eigvals
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (8,8) #(14.32,8)
# from PIL.Image import open, BILINEAR, fromarray

class AnySim():
	def __init__(self, test='FreeSpace', absorbing_boundaries=True, boundaries_width=16, wrap_correction='None', domain_decomp=True, cp=20):
		self.test = test 									# 'FreeSpace', '1D', '2D', OR '2D_low_contrast'
		self.absorbing_boundaries = absorbing_boundaries 	# True OR False
		self.boundaries_width = boundaries_width
		if self.absorbing_boundaries:
			self.bw_l = int(np.floor(self.boundaries_width))
			self.bw_r = int(np.ceil(self.boundaries_width))
		else:
			self.boundaries_width = 0
		self.wrap_correction = wrap_correction				# 'L_w', 'L_omega', OR 'L_corr'
		self.domain_decomp = domain_decomp					# Use domain decomposition (True by default) or not (False)
		self.cp = cp										# number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

		self.max_iters = int(1.e+5)# int(6.e+5)	# Maximum number of iterations
		self.iters = self.max_iters - 1
		self.N_dim = 1				# currently tackling only 1D problem, so number of dimensions = 1
		self.lambd = 1.				# Wavelength in um (micron)
		self.k0 = (1.*2.*np.pi)/(self.lambd)  # wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
		self.pixel_size = self.lambd/4
		self.N_roi = 256			# Num of points in ROI (Region of Interest)

		# set up coordinate ranges
		self.N = int(self.N_roi+2*self.boundaries_width)
		# self.x = np.arange(-self.boundaries_width, self.N - self.boundaries_width) * self.pixel_size
		# self.p = 2 * np.pi * np.fft.fftfreq(self.N, self.pixel_size)

		print(self.N, 'self.N before')
		self.N_domains = 2
		self.overlap = self.boundaries_width 	## Especially for the RAS_subdomain() method.
		# self.overlap = 1
		while (self.N-self.overlap)%self.N_domains != 0:
			self.N_roi += 1
			self.N = int(self.N_roi+2*self.boundaries_width)
		assert (self.N-self.overlap)%self.N_domains == 0
		print(self.N, 'self.N after')

		''' Create log folder / Check for existing log folder'''
		today = date.today()
		d1 = today.strftime("%Y%m%d")
		self.log_dir = 'logs/Logs_'+d1+'/'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		self.run_id = d1 + '_' + self.test
		if self.wrap_correction != 'None':
			self.run_id += '_' + self.wrap_correction
		if self.absorbing_boundaries:
			self.run_id += '_abs' + str(self.boundaries_width)

		self.run_loc = self.log_dir + self.run_id
		if not os.path.exists(self.run_loc):
			os.makedirs(self.run_loc)

		self.stats_file_name = self.log_dir + self.test + '_stats.txt'
		# print(self.stats_file_name)
		# exit(0)

	def runit(self):			# function that calls all the other 'main' functions
		s1 = time.time()
		self.init_setup()		# set up the grid, ROI, source, etc. based on the test
		self.print_details()	# print the simulation details

		Vraw = self.k0**2 * self.n**2
		Vraw = np.pad(Vraw, (self.boundaries_width, self.boundaries_width), mode='edge')

		if self.domain_decomp:
			self.n1 = int(np.ceil((self.N + self.overlap)/2))
			# nd = int((self.N-self.overlap)/self.N_domains + self.overlap)
			# n = [nd]*(self.N_domains)

			self.operators = []
			self.operators.append( self.make_operators(Vraw[:self.n1], 'left')[:2] )
			self.operators.append( self.make_operators(Vraw[-self.n1:], 'right')[:2] )
		else:
			self.medium, self.propagator, _, _ = self.make_operators(Vraw)	# Medium B = 1 - V, and Propagator (L+1)^(-1)

		# Scale the source term (and pad if boundaries)
		if self.absorbing_boundaries == True:
			self.b = self.Tl * np.pad(self.b, self.N_dim*((self.bw_l,self.bw_r),), mode='constant') # source term y
		else:
			self.b = self.Tl * self.b
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.alpha = 1.0						# ~step size of the Richardson iteration \in (0,1]
		if self.domain_decomp:
			''' Domain decomposition approaches '''
			# self.medium, self.propagator, B, L_plus_1_inv = self.make_operators(Vraw)	# Medium B = 1 - V, and Propagator (L+1)^(-1)
			# self.RAS_domain_decomp_n_iterate(B, L_plus_1_inv)	# (Restrictive) Additive Schwarz Method. Use global approximations as iterates
			self.RAS_subdomain()				# (Restrictive) Additive Schwarz Method. Use subdomain approximations as iterates
		else:
			self.iterate()						# single domain. No domain decomposition

		# Truncate u to ROI
		if self.absorbing_boundaries == True:
			self.u = self.u[self.bw_l:-self.bw_r]
			self.u_iter = self.u_iter[:, self.bw_l:-self.bw_r]

		# Print relative error between u and analytic solution (or Matlab result)
		if self.test == '1D':
			self.u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])

		self.rel_err = self.relative_error(self.u, self.u_true)
		print('\n')
		print('Relative error: {:.2e}'.format(self.rel_err))

		self.sim_time = time.time()-s1
		print('Simulation done (Time {} s)'.format(np.round(self.sim_time,2)))
		print('Saving stats and Plotting...')
		self.save_details()
		print('Saving stats done.')
		self.plot_details()	# Plot the final field, residual vs. iterations, and animation of field with iterations
		print('-'*50)
		return self.u

	def init_setup(self):
		## construct the refractive index map
		self.n = np.ones(self.N_roi)
		if self.test == '1D':
			self.n[99:130] = 1.5

		## define a point source
		source_amplitude = 1.
		self.b = np.zeros((self.N_roi,), dtype='complex_')
		self.b[0] = source_amplitude
		# self.b[int(self.N_roi/2)] = source_amplitude

		if self.test == 'FreeSpace':
			## Compare with the analytic solution
			x = np.arange(0,self.N_roi*self.pixel_size,self.pixel_size)
			x = np.pad(x, (64,64), mode='constant')
			h = self.pixel_size
			k = self.k0
			phi = k * x

			E_theory = 1.0j*h/(2*k) * np.exp(1.0j*phi) - h/(4*np.pi*k) * (np.exp(1.0j * phi) * ( np.exp(1.0j * (k-np.pi/h) * x) - np.exp(1.0j * (k+np.pi/h) * x)) - np.exp(-1.0j * phi) * ( -np.exp(-1.0j * (k-np.pi/h) * x) + np.exp(-1.0j * (k+np.pi/h) * x)))
			# special case for values close to 0
			small = np.abs(k*x) < 1.e-10
			E_theory[small] = 1.0j * h/(2*k) * (1 + 2j * np.arctanh(h*k/np.pi)/np.pi); # exact value at 0.
			self.u_true = E_theory[64:-64]

	def print_details(self):
		print('Test: \t\t\t', self.test)
		print('Absorbing boundaries: \t', self.absorbing_boundaries)
		print('Wrap correction: \t', self.wrap_correction)
		print('Boundaries width: \t', self.boundaries_width)

	# Make the operators: Medium B = 1 - V and Propagator (L+1)^(-1)
	def make_operators(self, Vraw, which_end=None):
		N = Vraw.shape[0]
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		if self.absorbing_boundaries:
			mu_min = 10.0/(self.boundaries_width * self.pixel_size)
		else:
			mu_min = 0
		mu_min = max( mu_min, 1.e+0/(N*self.pixel_size) )
		Vmin = np.imag( (self.k0 + 1j*np.max(mu_min))**2 )
		Vmax = 0.95
		V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 
		V0 = V0 - Vmin
		V = np.diag(-1j*(Vraw - V0))
		F = self.DFT_matrix(N)
		Finv = np.asarray(np.matrix(F).H/N)
		if self.wrap_correction == 'L_corr':
			# self.cp = 20
			p = 2*np.pi*np.fft.fftfreq(N, self.pixel_size)
			Lw_p = p**2
			Lw = Finv @ np.diag(Lw_p.flatten()) @ F
			L_corr = -np.real(Lw)                          # copy -Lw
			L_corr[:-self.cp,:-self.cp] = 0; L_corr[self.cp:,self.cp:] = 0  # Keep only upper and lower traingular corners of -Lw
			V = V + 1j*L_corr
		self.scaling = Vmax/self.checkV(V)
		V = self.scaling * V

		self.Tr = np.sqrt(self.scaling)
		self.Tl = 1j * self.Tr

		## Check that ||V|| < 1 (0.95 here)
		vc = self.checkV(V)
		if vc >= 1:
			raise Exception('||V|| not < 1, but {}'.format(vc))

		## B = 1 - V
		B = np.eye(N) - V
		if self.absorbing_boundaries:
			if which_end == None:
				N_roi = self.N_roi
			else:
				N_roi = N - self.boundaries_width
			np.fill_diagonal(B, self.pad_func(B.diagonal().copy(), N_roi, which_end))
		medium = lambda x: B @ x

		## Make Propagator (L+1)^(-1)
		if self.wrap_correction == 'L_omega':
			N_FastConv = N*10
		else:
			N_FastConv = N

		p = 2*np.pi*np.fft.fftfreq(N_FastConv, self.pixel_size)
		L_p = p**2
		L_p = 1j * self.scaling * (L_p - V0)
		Lp_inv = np.squeeze(1/(L_p+1))
		if self.wrap_correction == 'L_omega':
			# propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn( np.pad(x,(0,N_FastConv-N)) )))[:N]
			Ones = np.eye(N)
			Omega = np.zeros((N, N_FastConv))
			Omega[:,:N] = Ones
			F_Omega = self.DFT_matrix(N_FastConv)
			Finv_Omega = np.asarray(np.matrix(F_Omega).H/(N_FastConv))
			L_plus_1_inv = Omega @ Finv_Omega @ np.diag(Lp_inv.flatten()) @ F_Omega @ Omega.T
		else:
			# propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn(x)))
			L_plus_1_inv = Finv @ np.diag(Lp_inv.flatten()) @ F
		propagator = lambda x: L_plus_1_inv @ x

		## Check that A = L + V is accretive
		A = np.linalg.inv(L_plus_1_inv) - B
		acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.matrix(A).H))))
		if np.round(acc, 7) < 0:
			raise Exception('A is not accretive. ', acc)
		
		return [medium, propagator, B, L_plus_1_inv]

	# AnySim update
	def iterate(self):
		self.iter_step = 1
		residual = []
		u_iter = []
		for i in range(self.max_iters):
			t1 = self.medium(self.u) + self.b
			t1 = self.propagator(t1)
			t1 = self.medium(self.u-t1)       # residual
			if i==0:
				normb = np.linalg.norm(t1)
			nr = np.linalg.norm(t1)
			self.residual_i = nr/normb
			residual.append(self.residual_i)
			if self.residual_i < 1.e-6:
				self.iters = i
				print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(self.iters+1, self.residual_i))
				break
			self.u = self.u - (self.alpha * t1)
			u_iter.append(self.u)
		self.u = self.Tr * self.u

		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		self.residual = np.array(residual)

	''' Domain decomposition ''' ## (Restrictive) Additive Schwarz Method
	## Use global approximations as iterates
	def RAS_domain_decomp_n_iterate(self, B, L_plus_1_inv):

		# restriction operators
		R1 = np.zeros((self.n1,self.N))
		R2 = R1.copy()
		ones = np.eye(self.n1)
		R1[:,:self.n1] = ones
		R2[:,-self.n1:] = ones
		R = [R1, R2]

		# partition of unity
		D1 = np.diag( np.concatenate((np.ones(self.n1-self.overlap), 0.5*np.ones(self.overlap))) )
		D2 = np.diag( np.concatenate((0.5*np.ones(self.overlap), np.ones(self.n1-self.overlap))) )
		D = [D1, D2]

		A = (B) - (B @ L_plus_1_inv @ B)
		A1 = np.linalg.pinv(A[:self.n1,:self.n1].copy())
		A2 = np.linalg.pinv(A[-self.n1:,-self.n1:].copy())
		A_inv = [A1, A2]

		residual = []
		u_iter = []
		for i in range(self.max_iters):
			t1 = self.medium(self.u) + self.b
			t1 = self.propagator(t1)
			t1 = self.medium(t1 - self.u)       # residual

			# Compute and collect the residual for every iteration in a list and break iteration loop if min threshold hit
			if i==0:
				normb = np.linalg.norm(t1)
			nr = np.linalg.norm(t1)
			self.residual_i = nr/normb
			residual.append(self.residual_i)
			if self.residual_i < 1.e-6:
				self.iters = i
				print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(self.iters+1, self.residual_i))
				break

			# ## Additive Schwarz Method [More amenabla to theory]
			# for j in range(self.N_domains):
			# 	u[j] = R[j] @ self.u					# Restrict (using Rj) the global field self.u to the local field uj
			# 	u[j] = u[j] + A_inv[j] @ R[j] @ t1		# Update the local field uj. Restrict the global residual t1 (using Rj), solve locally (A_inv j) and then update
			# 	u[j] = R[j].T @ D[j] @ u[j]				# To assemble the global solution, couple the duplicated unknowns via a partition of unity
			# self.u = np.squeeze(np.array([sum(u)]))		# sum all local solutions from sub-domains to get the global solution

			### Restrictive Additive Schwarz Method (RAS Algorithm. Cai and Sarkis, 1999) [More efficient]
			Mras = np.zeros_like(self.u, dtype='complex_')
			for j in range(self.N_domains):
				'''	1. Restrict the global residual t1 using R[j]
					2. Solve locally using A_inv[j]
					3. Couple duplicated unknowns via a partition of unity using D[j]
					4. Extend back to global domain using R[j].T
					5. Add for all sub-domains Mras = Mras + ...
				'''
				Mras = Mras + R[j].T @ D[j] @ A_inv[j] @ R[j] @ t1
			self.u = self.u + Mras						# Update the global solution
			u_iter.append(self.u)
		self.u = self.Tr * self.u
		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		self.residual = np.array(residual)

	## Use subdomain approximations as iterates
	def RAS_subdomain(self):
		# restriction operators
		R1 = np.zeros((self.n1,self.N))
		R2 = R1.copy()
		ones = np.eye(self.n1)
		R1[:,:self.n1] = ones
		R2[:,-self.n1:] = ones
		R = [R1, R2]

		# partition of unity
		fnc_interp = lambda x: np.interp(np.arange(x), [0,x-1], [0,1])
		decay = fnc_interp(self.overlap)
		D1 = np.diag( np.concatenate((np.ones(self.n1-self.overlap), np.flip(decay))) )
		D2 = np.diag( np.concatenate((decay, np.ones(self.n1-self.overlap))) )
		D = [D1, D2]

		u = []
		b = []
		for j in range(self.N_domains):
			u.append(R[j]@self.u)
			b.append(R[j]@self.b)

		self.iter_step = 1
		tj = [0] * self.N_domains
		normb = [0] * self.N_domains
		residual_i = [1.0] * self.N_domains
		residual = [[0]] * self.N_domains
		u_iter = []
		breaker = False
		for i in range(self.max_iters):
			for j in range(self.N_domains):
				print('Iteration {}, sub-domain {}.'.format(i+1,j+1), end='\r')
				### Main update from here ---
				if i % self.iter_step == 0:
					u[j] = R[j] @ self.u
				tj[j] = self.operators[j][0](u[j]) + b[j]
				tj[j] = self.operators[j][1](tj[j])
				tj[j] = self.operators[j][0](u[j] - tj[j])       # subdomain residual
				u[j] = self.alpha * tj[j]
				if i % self.iter_step == 0:
					self.u = self.u - R[j].T @ D[j] @ u[j]
				### --- to here

					## Residual collection and checking				
					if i==0:
						normb[j] = np.linalg.norm(tj[j][self.boundaries_width:-self.overlap])
					nr = np.linalg.norm(tj[j][self.boundaries_width:-self.overlap])
					residual_i[j] = nr/normb[j]
					if i==0:
						residual[j] = [residual_i[j]]
					else:
						residual[j].append(residual_i[j])
					# if np.array([val < 1.e-6 for val in residual_i]).all():	## break only when BOTH subdomains' residual goes below threshold
					if residual_i[j] < 1.e-6: ## break when either domain's residual goes below threshold
						self.iters = i
						print('Stopping simulation at iter {}, sub-domain {}, residual {:.2e} <= 1.e-6'.format(self.iters+1, j+1, residual_i[j]))
						self.residual_i = residual_i[j]
						if j == 0:
							residual[1].append(np.nan)
						breaker = True
						break
			if breaker:
				break
			u_iter.append(self.u)
		self.u = self.Tr * self.u
		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		try:
			self.residual = np.array(residual)
			if self.residual.shape[0] < self.residual.shape[1]:
				self.residual = self.residual.T
		except:
			pass

	# Save details and stats
	def save_details(self):
		try:
			self.residual_i
		except:
			self.residual_i = np.nan
		with open(self.stats_file_name,'a') as fileopen:
			if self.wrap_correction == 'L_corr':
				fileopen.write('Test {}; absorbing boundaries {}; boundaries width {}; wrap correction {}; {:>2.2f} sec; {} iterations; final residual {:>2.2e}; relative error {:>2.2e}; corner points {} \n'.format(self.test, str(self.absorbing_boundaries), self.boundaries_width, self.wrap_correction, self.sim_time, self.iters+1, self.residual_i, self.rel_err, self.cp))
			else:
				fileopen.write('Test {}; absorbing boundaries {}; boundaries width {}; wrap correction {}; {:>2.2f} sec; {} iterations; final residual {:>2.2e}; relative error {:>2.2e} \n'.format(self.test, str(self.absorbing_boundaries), self.boundaries_width, self.wrap_correction, self.sim_time, self.iters+1, self.residual_i, self.rel_err))

	# Plotting functions
	def plot_details(self):
		self.x = np.arange(self.N_roi)*self.pixel_size
		if self.test == 'FreeSpace':
			self.label = 'Analytic solution'
		elif self.test == '1D':
			self.label = 'Matlab solution'

		self.plot_FieldNResidual()	# png
		# self.plot_field_final()	# png
		# self.plot_residual()		# png
		self.plot_field_iters()		# movie/animation/GIF
		plt.close('all')
		print('Plotting done.')

	def plot_FieldNResidual(self): # png
		plt.subplots(figsize=figsize, ncols=1, nrows=2)

		plt.subplot(2,1,1)
		if self.domain_decomp:
			plt.axvline(x=(self.N_roi-self.overlap)/2*self.pixel_size, c='b', ls='dashdot', lw=1.5)
			plt.axvline(x=(self.N_roi+self.overlap)/2*self.pixel_size, c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
		plt.plot(self.x, np.real(self.u_true), 'k', lw=2., label=self.label)
		plt.plot(self.x, np.real(self.u), 'r', lw=1., label='AnySim')
		plt.xlim([self.x[0]-self.x[1]*2,self.x[-1]+self.x[1]*2])
		plt.title('Field (Relative Error = {:.2e})'.format(self.rel_err))
		plt.ylabel('Amplitude')
		plt.xlabel('$x~[\lambda]$')
		plt.legend()
		plt.grid()

		plt.subplot(2,1,2)
		try:
			res_plots = plt.loglog(np.arange(1,self.iters+2, self.iter_step), self.residual, lw=1.5)
			if self.domain_decomp:
				plt.legend(iter(res_plots), ('Subdomain 1', 'Subdomain 2'))
		except:
			pass
		plt.axhline(y=1.e-6, c='k', ls=':')
		plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
		plt.ylim([0.8*np.nanmin(self.residual), 1.2*np.nanmax(self.residual)])
		plt.title('Residual. Iterations = {:.2e}'.format(self.iters+1))
		plt.ylabel('Residual')
		plt.xlabel('Iterations')
		plt.grid()

		title_text = ''
		if self.absorbing_boundaries:
			title_text = f'{title_text} Absorbing boundaries ({self.boundaries_width}). '
		if self.wrap_correction != 'None':
			title_text = f'{title_text} Wrap correction: {self.wrap_correction}. '
		plt.suptitle(title_text)

		plt.tight_layout()
		if self.wrap_correction == 'L_corr':
			plt.savefig(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual_cp{self.cp}.png', bbox_inches='tight', pad_inches=0.03, dpi=100)
		else:
			plt.savefig(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual.png', bbox_inches='tight', pad_inches=0.03, dpi=100)
		# plt.draw()
		plt.close()

	def plot_field_iters(self): # movie/animation/GIF
		self.u_iter = np.real(self.u_iter)

		fig = plt.figure(figsize=(14.32,8))
		if self.domain_decomp:
			plt.axvline(x=(self.N_roi-self.overlap)/2*self.pixel_size, c='b', ls='dashdot', lw=1.5)
			plt.axvline(x=(self.N_roi+self.overlap)/2*self.pixel_size, c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
		plt.plot(self.x, np.real(self.u_true), 'k', lw=0.75, label=self.label)
		plt.xlabel("$x$")
		plt.ylabel("Amplitude")
		plt.xlim([self.x[0]-self.x[1]*2,self.x[-1]+self.x[1]*2])
		# plt.ylim([np.min(self.u_iter), np.max(self.u_iter)])
		plt.grid()
		line, = plt.plot([] , [], 'r', lw=1., animated=True, label='AnySim')
		line.set_xdata(self.x)
		title = plt.title('')
		plt.legend()

		# Plot 100 or fewer frames. Takes much longer for any more frames.
		if self.iters > 100:
			plot_iters = 100
			iters_trunc = np.linspace(0,self.iters-1,plot_iters).astype(int)
			u_iter_trunc = self.u_iter[iters_trunc]
		else:
			plot_iters = self.iters
			iters_trunc = np.arange(self.iters)
			u_iter_trunc = self.u_iter

		def animate(i):
			line.set_ydata(u_iter_trunc[i])		# update the data.
			title_text = f'Iteration {iters_trunc[i]+1}. '
			if self.absorbing_boundaries:
				title_text = f'{title_text} Absorbing boundaries ({self.boundaries_width}). '
			if self.wrap_correction != 'None':
				title_text = f'{title_text} Wrap correction: {self.wrap_correction}. '
			title.set_text(title_text)
			return line, title,
		ani = animation.FuncAnimation(
			fig, animate, interval=100, blit=True, frames=plot_iters)
		writer = animation.FFMpegWriter(
		    fps=10, metadata=dict(artist='Me'))
		if self.wrap_correction == 'L_corr':
			ani.save(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_Field_cp{self.cp}.mp4', writer=writer)
		else:
			ani.save(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_Field.mp4', writer=writer)
		plt.close()

	## pad boundaries
	def pad_func(self, M, M_roi, which_end=None):
		# boundary_ = lambda x: (np.arange(1,x+1)-0.21).T/(x+0.66)
		boundary_ = lambda x: np.interp(np.arange(x), [0,x-1], [0.04981993,0.95018007])
		left_boundary = boundary_(np.floor(self.boundaries_width))
		right_boundary = boundary_(np.ceil(self.boundaries_width))
		if which_end == None:
			full_filter = np.concatenate((left_boundary, np.ones((M_roi,)), np.flip(right_boundary)))
		elif which_end == 'left':
			full_filter = np.concatenate((left_boundary, np.ones((M_roi,))))
		elif which_end == 'right':
			full_filter = np.concatenate((np.ones((M_roi,)), np.flip(right_boundary)))
		M = M * full_filter
		return M

	def boundaries_window(self, L):
		x = np.expand_dims(np.arange(L)/(L-1), axis=1)
		a2 = np.expand_dims(np.array([-0.4891775, 0.1365995/2, -0.0106411/3]) / (0.3635819 * 2 * np.pi), axis=1)
		return np.squeeze(np.sin(x * np.expand_dims(np.array([1, 2, 3]), axis=0) * 2 * np.pi) @ a2 + x)

	## DFT matrix
	def DFT_matrix(self, N):
		l, m = np.meshgrid(np.arange(N), np.arange(N))
		omega = np.exp( - 2 * np.pi * 1j / N )
		return np.power( omega, l * m )

	## Relative error
	def relative_error(self, E_, E_true):
		return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )

	## Spectral radius
	def checkV(self, A):
		return np.linalg.norm(A,2)


s1 = time.time()
anysim = AnySim()
anysim.runit()
e1 = time.time() - s1
print('Total time (including plotting): ', np.round(e1,2))
print('-'*50)

# s2 = time.time()
# anysim = AnySim(wrap_correction='L_omega')
# anysim.runit()
# e2 = time.time() - s2
# print('Total time (including plotting): ', np.round(e2,2))
# print('-'*50)

# s3 = time.time()
# anysim = AnySim(wrap_correction='L_corr')
# anysim.runit()
# e3 = time.time() - s3
# print('Total time (including plotting): ', np.round(e3,2))
# print('-'*50)

print('Done')