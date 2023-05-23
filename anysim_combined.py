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
	def __init__(self, test='FreeSpace', absorbing_boundaries=True, boundaries_width=16, wrap_correction='None', cp=20):
		self.test = test 									# 'FreeSpace', '1D', '2D', OR '2D_low_contrast'
		self.absorbing_boundaries = absorbing_boundaries 	# True OR False
		self.boundaries_width = boundaries_width
		if self.absorbing_boundaries:
			self.bw_l = int(np.floor(self.boundaries_width))
			self.bw_r = int(np.ceil(self.boundaries_width))
		else:
			self.boundaries_width = 0
		self.wrap_correction = wrap_correction				# 'L_w', 'L_omega', OR 'L_corr'
		self.cp = cp										# number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

		self.max_iters = 2100# int(6.e+5)	# Maximum number of iterations
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
		self.overlap = 1
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
		self.make_medium()		# B = 1 - V
		self.make_propagator()	# L and (L+1)^(-1)

		# Scale the source term (and pad if boundaries)
		if self.absorbing_boundaries == True:
			self.b = self.Tl * np.pad(self.b, self.N_dim*((self.bw_l,self.bw_r),), mode='constant') # source term y
		else:
			self.b = self.Tl * self.b
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.iterate()		# single domain

		''' Domain decomposition approaches '''
		# self.RAS_domain_decomp_n_iterate()			# (Restrictive) Additive Schwartz Method. Use global approximations as iterates
		# self.GLOBAL_domain_decomp_n_iterate()		# 2019 Gander and Zhang, Algorithm 4. Use global approximations as iterates
		# self.SUBDOMAIN_domain_decomp_n_iterate()	# 2019 Gander and Zhang, Algorithm 2. Use subdomain approximations as iterates
		
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

	# Medium. B = 1 - V
	def make_medium(self):
		Vraw = self.k0**2 * self.n**2
		Vraw = np.pad(Vraw, (self.boundaries_width, self.boundaries_width), mode='edge')
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		if self.absorbing_boundaries:
			mu_min = 10.0/(self.boundaries_width * self.pixel_size)
		else:
			mu_min = 0
		mu_min = max( mu_min, 1.e+0/(self.N*self.pixel_size) )
		Vmin = np.imag( (self.k0 + 1j*np.max(mu_min))**2 )
		Vmax = 0.95
		self.V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 
		self.V0 = self.V0 - Vmin
		self.V = np.diag(-1j*(Vraw - self.V0))
		self.F = self.DFT_matrix(self.N)
		self.Finv = np.asarray(np.matrix(self.F).H/self.N)
		if self.wrap_correction == 'L_corr':
			# self.cp = 20
			p = 2*np.pi*np.fft.fftfreq(self.N, self.pixel_size)
			Lw_p = p**2
			Lw = self.Finv @ np.diag(Lw_p.flatten()) @ self.F
			L_corr = -np.real(Lw)                          # copy -Lw
			L_corr[:-self.cp,:-self.cp] = 0; L_corr[self.cp:,self.cp:] = 0  # Keep only upper and lower traingular corners of -Lw
			self.V = self.V + 1j*L_corr
		self.scaling = Vmax/self.checkV(self.V)
		self.V = self.scaling * self.V

		self.Tr = np.sqrt(self.scaling)
		self.Tl = 1j * self.Tr

		## Check that ||V|| < 1 (0.95 here)
		vc = self.checkV(self.V)
		if vc >= 1:
			raise Exception('||V|| not < 1, but {}'.format(vc))

		## B = 1 - V
		self.B = np.eye(self.N) - self.V
		if self.absorbing_boundaries:
			np.fill_diagonal(self.B, self.pad_func(self.B.diagonal().copy(), self.N_roi))
		self.medium = lambda x: self.B @ x

	# Propagator. (L+1)^(-1)
	def make_propagator(self):
		if self.wrap_correction == 'L_omega':
			N = self.N*10
		else:
			N = self.N

		p = 2*np.pi*np.fft.fftfreq(N, self.pixel_size)
		L_p = p**2
		L_p = 1j * self.scaling * (L_p - self.V0)
		Lp_inv = np.squeeze(1/(L_p+1))
		if self.wrap_correction == 'L_omega':
			# self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn( np.pad(x,(0,N-self.N)) )))[:self.N]
			Ones = np.eye(self.N)
			Omega = np.zeros((self.N, N))
			Omega[:,:self.N] = Ones
			F_Omega = self.DFT_matrix(N)
			Finv_Omega = np.asarray(np.matrix(F_Omega).H/(N))
			self.L_plus_1_inv = Omega @ Finv_Omega @ np.diag(Lp_inv.flatten()) @ F_Omega @ Omega.T
		else:
			# self.propagator = lambda x: (np.fft.ifftn(Lp_inv * np.fft.fftn(x)))
			self.L_plus_1_inv = self.Finv @ np.diag(Lp_inv.flatten()) @ self.F
		self.propagator = lambda x: self.L_plus_1_inv @ x

		## Check that A = L + V is accretive
		A = np.linalg.inv(self.L_plus_1_inv) - self.B
		acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.matrix(A).H))))
		if np.round(acc, 7) < 0:
			raise Exception('A is not accretive. ', acc)

	# AnySim update
	def iterate(self):
		self.alpha = 0.75									# ~step size of the Richardson iteration \in (0,1]
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

	''' Domain decomposition '''
	# (Restrictive) Additive Schwartz Method
	def RAS_domain_decomp_n_iterate(self):
		n1 = int(np.ceil((self.N + self.overlap)/2))
		n2 = n1

		# restriction operators
		R1 = np.zeros((n1,self.N))
		ones = np.eye(n1)
		R1[:,:n1] = ones
		R2 = np.zeros((n2,self.N))
		R2[:,-n2:] = ones
		R = [R1, R2]

		# partition of unity
		D1 = np.diag( np.concatenate((np.ones(n1-self.overlap), 0.5*np.ones(self.overlap))) )
		D2 = np.diag( np.concatenate((0.5*np.ones(self.overlap), np.ones(n1-self.overlap))) )
		D = [D1, D2]

		A = (self.B) - (self.B @ self.L_plus_1_inv @ self.B)
		A1 = np.linalg.pinv(A[:n1,:n1].copy())
		A2 = np.linalg.pinv(A[-n2:,-n2:].copy())
		A_inv = [A1, A2]

		v1 = np.zeros((n1,));	v2 = np.zeros((n2,))
		v = [v1, v2]

		residual = []
		u_iter = []
		u = [None] * self.N_domains
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

			# ## Additive Schwartz Method [More amenabla to theory]
			# for j in range(self.N_domains):
			# 	u[j] = R[j] @ self.u					# Restrict (using Rj) the global field self.u to the local field uj
			# 	u[j] = u[j] + A_inv[j] @ R[j] @ t1		# Update the local field uj. Restrict the global residual t1 (using Rj), solve locally (A_inv j) and then update
			# 	u[j] = R[j].T @ D[j] @ u[j]				# To assemble the global solution, couple the duplicated unknowns via a partition of unity
			# self.u = np.squeeze(np.array([sum(u)]))		# sum all local solutions from sub-domains to get the global solution

			### Restrictive Additive Schwartz Method (RAS Algorithm. Cai and Sarkis, 1999) [More efficient]
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

	# 2019 Gander and Zhang, Algorithm 4. Use global approximations as iterates
	def GLOBAL_domain_decomp_n_iterate(self):
		nd = int((self.N-self.overlap)/self.N_domains + self.overlap)
		n = [nd]*(self.N_domains)
		A = (self.B) - (self.B @ self.L_plus_1_inv @ self.B)

		Q_l = []
		Q_r = []
		phi_f = []
		phi_b = []
		v = []
		R = []
		A_inv = []
		for j in range(self.N_domains):
			Q_l.append( np.diag( np.concatenate((0.5*np.ones((self.overlap)), np.zeros(n[j]-self.overlap)))) )
			Q_r.append( np.diag( np.concatenate((np.zeros(n[j]-self.overlap), 0.5*np.ones((self.overlap))))) )
			v.append( np.zeros((n[j],)) )

			Rj = np.zeros((n[j], self.N))
			ones = np.eye(n[j])
			if j==0:
				Rj[:,:n[j]] = ones
				A_inv.append( np.linalg.pinv(A[:n[j],:n[j]]) )
				phi_f.append( np.diag( np.ones(n[j]) ) )
				phi_b.append( np.diag( np.concatenate((np.ones(n[j]-self.overlap), np.zeros((self.overlap)))) ) )
			elif j==self.N_domains-1:
				Rj[:,-n[j]:] = ones
				A_inv.append( np.linalg.pinv(A[-n[j]:,-n[j]:]) )
				phi_f.append( np.diag( np.concatenate((np.zeros((self.overlap)), np.ones(n[j]-self.overlap))) ) )
				phi_b.append( np.diag( np.ones(n[j]) ) )
			else:
				j_start = j*(n[j-1]-self.overlap)
				j_end = j_start+n[j]
				Rj[:,j_start:j_end] = ones
				A_inv.append( np.linalg.pinv(A[j_start:j_end,j_start:j_end]) )
				phi_f.append( np.diag( np.concatenate((np.zeros((self.overlap)), np.ones(n[j]-self.overlap))) ) )
				phi_b.append( np.diag( np.concatenate((np.ones(n[j]-self.overlap), np.zeros((self.overlap)))) ) )
			R.append( Rj )

		P_l = Q_l.copy()
		P_r = Q_r.copy()

		assert ( np.sum(np.array([R[j].T @ phi_f[j] @ R[j] for j in range(self.N_domains)]), axis=0) == np.eye(self.N) ).all()
		assert ( np.sum(np.array([R[j].T @ phi_b[j] @ R[j] for j in range(self.N_domains)]), axis=0) == np.eye(self.N) ).all()

		assert all([np.linalg.norm(R[j].T @ phi_f[j] @ R[j] @ R[l].T @ phi_f[l]) == 0 for j in range(self.N_domains) for l in range(self.N_domains) if j != l])
		assert all([np.linalg.norm(R[j].T @ phi_b[j] @ R[j] @ R[l].T @ phi_b[l]) == 0 for j in range(self.N_domains) for l in range(self.N_domains) if j != l])

		row_1l = []
		row_2A = []
		row_3r = []
		for j in range(self.N_domains):
			row_1l.append( np.linalg.pinv(Q_l[j] @ A_inv[j] + P_l[j]) @ Q_l[j] @ R[j] )
			row_2A.append( A_inv[j] @ R[j] )
			row_3r.append( np.linalg.pinv(Q_r[j] @ A_inv[j] + P_r[j]) @ Q_r[j] @ R[j] )

		residual = []
		u_iter = []

		### Double sweep Optimized Schwartz Method (DOSM) -- in the global correction form
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

			### Double sweep Optimized Schwartz Method (DOSM) -- in the global correction form
			for j in range(self.N_domains):
				print('sub-domain', j+1, end='\r')
				v[j] = row_1l[j] @ t1
				v[j] = v[j] + row_2A[j] @ t1
				v[j] = v[j] + row_3r[j] @ t1
				self.u = self.u + R[j].T @ phi_f[j] @ v[j]
				u_iter.append(self.u)

		self.u = self.Tr * self.u
		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		self.residual = np.array(residual)

	# 2019 Gander and Zhang, Algorithm 2. Use subdomain approximations as iterates
	def SUBDOMAIN_domain_decomp_n_iterate(self):
		nd = int((self.N-self.overlap)/self.N_domains + self.overlap)
		n = [nd]*(self.N_domains)
		A = (self.B) - (self.B @ self.L_plus_1_inv @ self.B)

		Q_l = []
		Q_r = []
		R = []
		A_inv = []
		
		Vraw = self.k0**2 * self.n**2
		Vraw = np.pad(Vraw, (self.boundaries_width, self.boundaries_width), mode='edge')
		medium_propagator = []
		
		for j in range(self.N_domains):
			Q_l.append( np.diag( np.concatenate((0.5*np.ones((self.overlap)), np.zeros(n[j]-self.overlap)))) )
			Q_r.append( np.diag( np.concatenate((np.zeros(n[j]-self.overlap), 0.5*np.ones((self.overlap))))) )

			Rj = np.zeros((n[j], self.N))
			ones = np.eye(n[j])
			if j==0:
				Rj[:,:n[j]] = ones
				A_inv.append( np.linalg.pinv(A[:n[j],:n[j]]) )
				medium_propagator.append( self.make_separate_medium_propagator(Vraw[:n[j]], 'left') )
			elif j==self.N_domains-1:
				Rj[:,-n[j]:] = ones
				A_inv.append( np.linalg.pinv(A[-n[j]:,-n[j]:]) )
				medium_propagator.append( self.make_separate_medium_propagator(Vraw[-n[j]:], 'right') )
			else:
				j_start = j*(n[j-1]-self.overlap)
				j_end = j_start+n[j]
				Rj[:,j_start:j_end] = ones
				A_inv.append( np.linalg.pinv(A[j_start:j_end,j_start:j_end]) )
				medium_propagator.append( self.make_separate_medium_propagator(Vraw[j_start:j_end], 'NoPad') )
			R.append( Rj )

		P_l = Q_l.copy()
		P_r = Q_r.copy()

		row_1l = []
		row_3r = []
		for j in range(self.N_domains):
			row_1l.append( np.linalg.pinv(Q_l[j] @ A_inv[j] + P_l[j]) )
			row_3r.append( np.linalg.pinv(Q_r[j] @ A_inv[j] + P_r[j]) )

		residual = []
		u_iter = []
		tj = [None] * self.N_domains

		### Double sweep Optimized Schwartz Method (DOSM) -- in the subdomain transmission form
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
				print('\n')
				print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(self.iters+1, self.residual_i))
				break

			for j in range(self.N_domains):
				print('Iteration {}, sub-domain {}'.format(i+1,j+1), end='\r')

				j_start = j*(n[j-1]-self.overlap)
				j_end = j_start+n[j]

				tj[j] = medium_propagator[j][0](self.u[j_start:j_end]) + self.b[j_start:j_end]
				tj[j] = medium_propagator[j][1](tj[j])
				tj[j] = medium_propagator[j][0](tj[j] - self.u[j_start:j_end])       # subdomain residual

				self.u[j_start:j_end] = row_1l[j] @ ( (Q_l[j] @ tj[j]) + (P_l[j] @ self.u[j_start:j_end]) )
				self.u[j_start:j_end] += A_inv[j] @ self.b[j_start:j_end]
				self.u[j_start:j_end] += row_3r[j] @ ( (Q_r[j] @ tj[j]) + (P_r[j] @ self.u[j_start:j_end]) )

				# self.u[j_start:j_end] = row_1l[j] @ ( (Q_l[j] @ t1[j_start:j_end]) + (P_l[j] @ self.u[j_start:j_end]) )
				# self.u[j_start:j_end] += A_inv[j] @ self.b[j_start:j_end]
				# self.u[j_start:j_end] += row_3r[j] @ ( (Q_r[j] @ t1[j_start:j_end]) + (P_r[j] @ self.u[j_start:j_end]) )

				u_iter.append(self.u)

			for j in reversed(range(self.N_domains)):
				print('Iteration {}, sub-domain {}'.format(i+1,j+1), end='\r')

				j_start = j*(n[j-1]-self.overlap)
				j_end = j_start+n[j]

				tj[j] = medium_propagator[j][0](self.u[j_start:j_end]) + self.b[j_start:j_end]
				tj[j] = medium_propagator[j][1](tj[j])
				tj[j] = medium_propagator[j][0](tj[j] - self.u[j_start:j_end])       # subdomain residual

				self.u[j_start:j_end] += row_1l[j] @ ( (Q_l[j] @ tj[j]) + (P_l[j] @ self.u[j_start:j_end]) )
				self.u[j_start:j_end] += A_inv[j] @ self.b[j_start:j_end]
				self.u[j_start:j_end] += row_3r[j] @ ( (Q_r[j] @ tj[j]) + (P_r[j] @ self.u[j_start:j_end]) )

				u_iter.append(self.u)


		self.u = self.Tr * self.u
		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		self.residual = np.array(residual)

	# Construct the split B and 1/(L+1) operators for each subdomain
	def make_separate_medium_propagator(self, Vraw, which_end=None):
		N = Vraw.shape[0]
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		if self.absorbing_boundaries:
			mu_min = 10.0/(self.boundaries_width * self.pixel_size)
		else:
			mu_min = 0
		mu_min = max( mu_min, 1.e+0/(self.N*self.pixel_size) )
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
		scaling = Vmax/self.checkV(V)
		V = scaling * V

		Tr = np.sqrt(scaling)
		Tl = 1j * Tr

		## Check that ||V|| < 1 (0.95 here)
		vc = self.checkV(V)
		if vc >= 1:
			raise Exception('||V|| not < 1, but {}'.format(vc))

		## B = 1 - V
		B = np.eye(N) - V
		if self.absorbing_boundaries:
			if which_end != 'NoPad':
				np.fill_diagonal(B, self.pad_func(B.diagonal().copy(), Vraw[self.boundaries_width:].size, which_end))
		medium = lambda x: B @ x

		p = 2*np.pi*np.fft.fftfreq(N, self.pixel_size)
		L_p = p**2
		L_p = 1j * self.scaling * (L_p - self.V0)
		Lp_inv = np.squeeze(1/(L_p+1))
		L_plus_1_inv = Finv @ np.diag(Lp_inv.flatten()) @ F
		propagator = lambda x: L_plus_1_inv @ x
		
		## Check that A = L + V is accretive
		A = np.linalg.inv(L_plus_1_inv) - B
		acc = np.min(np.real(np.linalg.eigvals(A + np.asarray(np.matrix(A).H))))
		if np.round(acc, 7) < 0:
			raise Exception('A is not accretive. ', acc)
		
		return [medium, propagator]


	# Save details and stats
	def save_details(self):
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
		print('Plotting done.')

	def plot_FieldNResidual(self): # png
		plt.subplots(figsize=figsize, ncols=1, nrows=2)

		plt.subplot(2,1,1)
		plt.plot(self.x, np.real(self.u_true), 'k', lw=2., label=self.label)
		plt.plot(self.x, np.real(self.u), 'r', lw=1., label='RelErr = {:.2e}'.format(self.rel_err))
		plt.title('Field')
		plt.ylabel('Amplitude')
		plt.xlabel('$x~[\lambda]$')
		plt.legend()
		plt.grid()

		plt.subplot(2,1,2)
		plt.loglog(np.arange(1,self.iters+2), self.residual, 'k', lw=1.5)
		plt.axhline(y=1.e-6, c='k', ls=':')
		plt.yticks([1.e+0, 1.e-2, 1.e-4, 1.e-6])
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
		plt.plot(self.x, np.real(self.u_true), 'k:', lw=0.75, label=self.label)
		plt.xlabel("$x$")
		plt.ylabel("Amplitude")
		plt.xlim([self.x[0]-self.x[1]*2,self.x[-1]+self.x[1]*2])
		# plt.ylim([np.min(self.u_iter), np.max(self.u_iter)])
		plt.grid()
		line, = plt.plot([] , [], 'b', lw=1., animated=True)
		line.set_xdata(self.x)
		title = plt.title('')

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
		# left_boundary_linear = boundary_(np.floor(self.boundaries_width))
		# right_boundary_linear = boundary_(np.ceil(self.boundaries_width))
		left_boundary = self.boundaries_window(np.floor(self.boundaries_width))
		right_boundary = self.boundaries_window(np.ceil(self.boundaries_width))
		# full_filter = np.concatenate((left_boundary, np.ones((self.N_roi,)), np.flip(right_boundary)))
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



# sim = AnySim()
# sim.runit()
