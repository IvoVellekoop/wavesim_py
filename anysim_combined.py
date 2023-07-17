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
	def __init__(self, test='custom', lambd=1, ppw=4, boundaries_width=20, N=256, n=[0], source_amplitude=1., source_location=0, N_domains=2, overlap=20, wrap_correction='None', cp=20):
		self.test = test 									# 'FreeSpace', '1D', '2D', OR '2D_low_contrast'

		self.absorbing_boundaries = True 	# True OR False
		self.boundaries_width = boundaries_width
		if self.absorbing_boundaries:
			self.bw_l = int(np.floor(self.boundaries_width))
			self.bw_r = int(np.ceil(self.boundaries_width))
		else:
			self.boundaries_width = 0

		self.lambd = 1.							# Wavelength in um (micron)
		self.ppw = ppw							# points per wavelength
		self.k0 = (1.*2.*np.pi)/(self.lambd)	# wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
		self.pixel_size = self.lambd/self.ppw
		self.N_roi_orig = N
		self.N_roi = N# 256						# Num of points in ROI (Region of Interest)
		self.N = int(self.N_roi+2*self.boundaries_width)

		self.N_domains = N_domains
		self.overlap = overlap# self.boundaries_width
		while (self.N-self.overlap)%self.N_domains != 0:
			self.N_roi += 1
			self.N = int(self.N_roi+2*self.boundaries_width)
		assert (self.N-self.overlap)%self.N_domains == 0

		self.source_amplitude = source_amplitude
		self.source_location = source_location

		self.b = np.zeros((self.N_roi,), dtype='complex_')
		self.b[self.source_location] = self.source_amplitude

		self.n = n
		## construct the refractive index map if needed
		if self.test != 'custom':
			self.n = np.ones(self.N_roi)
			if self.test == '1D':
				self.n[99:130] = 1.5
		self.N_dim = np.ndim(self.n)# 1		# currently tackling only 1D problem, so number of dimensions = 1

		self.wrap_correction = wrap_correction	# 'L_w', 'L_omega', OR 'L_corr'
		self.cp = cp							# number of corner points (c.p.) in the upper and lower triangular corners of the L_corr matrix

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
		self.run_id += '_abs' + str(self.boundaries_width)
		if self.wrap_correction != 'None':
			self.run_id += '_' + self.wrap_correction
		self.run_id += '_Ndoms' + str(self.N_domains)

		self.stats_file_name = self.log_dir + self.test + '_stats.txt'

	def runit(self):			# function that calls all the other 'main' functions
		s1 = time.time()
		self.print_details()	# print the simulation details

		# Make operators: Medium B = 1 - V, and Propagator (L+1)^(-1)
		Vraw = self.k0**2 * self.n**2
		Vraw = np.pad(Vraw, (self.boundaries_width, self.boundaries_width), mode='edge')

		self.n1 = int((self.N-self.overlap)/self.N_domains + self.overlap)
		self.operators = []
		if self.N_domains==1:
			self.operators.append( self.make_operators(Vraw) )
		else:
			self.operators.append( self.make_operators(Vraw[:self.n1], 'left') )
			for i in range(1,self.N_domains-1):
				self.operators.append( self.make_operators(Vraw[i*(self.n1-self.overlap):i*(self.n1-self.overlap)+self.n1], None) )
			self.operators.append( self.make_operators(Vraw[-self.n1:], 'right') )

		# Scale the source term (and pad if boundaries)
		if self.absorbing_boundaries:
			self.b = self.Tl * np.pad(self.b, self.N_dim*((self.bw_l,self.bw_r),), mode='constant') # source term y
		else:
			self.b = self.Tl * self.b
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.max_iters = int(2.e+3)		# Maximum number of iterations
		self.iters = self.max_iters - 1
		self.alpha = 0.9				# ~step size of the Richardson iteration \in (0,1]
		self.threshold_residual = 1.e-6
		self.iter_step = 1
		self.iterate()

		# Truncate u to ROI
		if self.absorbing_boundaries:
			self.u = self.u[self.bw_l:-self.bw_r]
			self.u_iter = self.u_iter[:, self.bw_l:-self.bw_r]

		# Print relative error between u and analytic solution (or Matlab result)
		if self.test == 'FreeSpace' or self.test == 'custom':
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
		elif self.test == '1D':
			self.u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])

		if self.u_true.shape[0] != self.N_roi:
			self.N_roi = self.N_roi_orig
			self.u = self.u[:self.N_roi_orig]
			self.u_iter = self.u_iter[:, :self.N_roi_orig]

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

	def print_details(self):
		print(f'{self.N_dim} dimensional problem')
		if self.test != 'custom':
			print('Test: \t\t\t', self.test)
		if self.wrap_correction != 'None':
			print('Wrap correction: \t', self.wrap_correction)
		print('Boundaries width: \t', self.boundaries_width)
		if self.N_domains > 1:
			print(f'Decomposing into {self.N_domains} domains, with overlap {self.overlap}')

	# Make the operators: Medium B = 1 - V and Propagator (L+1)^(-1)
	def make_operators(self, Vraw, which_end='Both'):
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
			p = 2*np.pi*np.fft.fftfreq(N, self.pixel_size)
			Lw_p = p**2
			Lw = Finv @ np.diag(Lw_p.flatten()) @ F
			L_corr = -np.real(Lw)
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
		if self.absorbing_boundaries and which_end != None:
			if which_end == 'Both':
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
		
		return [medium, propagator]

	# AnySim update
	def iterate(self):
		# restriction operator(s)
		ones = np.eye(self.n1)
		R = []
		if self.N_domains==1:
			R.append(ones)
			D = R.copy()
		else:
			R_0 = np.zeros((self.n1,self.N))
			for i in range(self.N_domains):
				R_mid = R_0.copy()
				R_mid[:,i*(self.n1-self.overlap):i*(self.n1-self.overlap)+self.n1] = ones
				R.append(R_mid)

			# partition of unity
			fnc_interp = lambda x: np.interp(np.arange(x), [0,x-1], [0,1])
			decay = fnc_interp(self.overlap)
			D = []
			D1 = np.diag( np.concatenate((np.ones(self.n1-self.overlap), np.flip(decay))) )
			D.append(D1)
			D_mid = np.diag( np.concatenate((decay, np.ones(self.n1-2*self.overlap), np.flip(decay))) )
			for _ in range(1,self.N_domains-1):
				D.append(D_mid)
			D_end = np.diag( np.concatenate((decay, np.ones(self.n1-self.overlap))) )
			D.append(D_end)

		u = []
		b = []
		for j in range(self.N_domains):
			u.append(R[j]@self.u)
			b.append(R[j]@self.b)

		tj = [0] * self.N_domains
		normb = [0] * self.N_domains
		residual_i = [1.0] * self.N_domains
		residual = [[0]] * self.N_domains
		u_iter = []
		breaker = False
		for i in range(self.max_iters):
			for j in range(self.N_domains):
				print('Iteration {}, sub-domain {}.'.format(i+1,j+1), end='\r')
				### Main update START ---
				# if i % self.iter_step == 0:
				u[j] = R[j] @ self.u
				tj[j] = self.operators[j][0](u[j]) + b[j]
				tj[j] = self.operators[j][1](tj[j])
				tj[j] = self.operators[j][0](u[j] - tj[j])       # subdomain residual
				### --- continued below ---

				## Residual collection and checking				
				if i==0:
					if j==0:
						normb[j] = np.linalg.norm(tj[j][self.boundaries_width:-self.overlap])
					elif j==self.N_domains-1:
						normb[j] = np.linalg.norm(tj[j][self.overlap:-self.boundaries_width])
					else:
						normb[j] = np.linalg.norm(tj[j][self.overlap:-self.overlap])
				if j==0:
					nr = np.linalg.norm(tj[j][self.boundaries_width:-self.overlap])
				elif j==self.N_domains-1:
					nr = np.linalg.norm(tj[j][self.overlap:-self.boundaries_width])
				else:
					nr = np.linalg.norm(tj[j][self.overlap:-self.overlap])
				residual_i[j] = nr/normb[j]
				if i==0:
					residual[j] = [residual_i[j]]
				else:
					residual[j].append(residual_i[j])

				# if np.array([val < self.threshold_residual for val in residual_i]).all():	## break only when ALL subdomains' residual goes below threshold
				if residual_i[j] < self.threshold_residual: ## break when any domain's residual goes below threshold
					self.iters = i
					print('Stopping simulation at iter {}, sub-domain {}, residual {:.2e} <= {}'.format(self.iters+1, j+1, residual_i[j], self.threshold_residual))
					self.residual_i = residual_i[j]
					for m in range(1,self.N_domains):
						while len(residual[m]) != len(residual[0]):
							residual[m].append(np.nan)
					breaker = True
					break

				### --- continued below ---
				u[j] = self.alpha * tj[j]
				# if i % self.iter_step == 0:
				self.u = self.u - R[j].T @ D[j] @ u[j]
				### Main update END ---
			if breaker:
				break
			u_iter.append(self.u)
		self.u = self.Tr * self.u
		self.u_iter = self.Tr.flatten() * np.array(u_iter)

		# residual[1] = residual[1][::2]	# if update order is 0-1-2-1-0-... (i.e., 1 is repeated twice in one global iteration)
		self.residual = np.array(residual)
		if self.residual.shape[0] < self.residual.shape[1]:
			self.residual = self.residual.T

	# Save details and stats
	def save_details(self):
		try:
			self.residual_i
		except:
			self.residual_i = np.nan
		with open(self.stats_file_name,'a') as fileopen:
			fileopen.write('Test {}; absorbing boundaries {}; boundaries width {}; N_domains {}; overlap {}; wrap correction {}; corner points {}; {:>2.2f} sec; {} iterations; final residual {:>2.2e}; relative error {:>2.2e} \n'.format(self.test, str(self.absorbing_boundaries), self.boundaries_width, self.N_domains, self.overlap, self.wrap_correction, self.cp, self.sim_time, self.iters+1, self.residual_i, self.rel_err))

	# Plotting functions
	def plot_details(self):
		self.x = np.arange(self.N_roi)*self.pixel_size
		if self.test == 'FreeSpace' or self.test == 'custom':
			self.label = 'Analytic solution'
		elif self.test == '1D':
			self.label = 'Matlab solution'

		self.plot_FieldNResidual()	# png
		self.plot_field_iters()		# movie/animation/GIF
		plt.close('all')
		print('Plotting done.')

	def plot_basics(self, plt):
		if self.N_domains > 1:
			plt.axvline(x=(self.n1-self.boundaries_width-self.overlap)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
			plt.axvline(x=(self.n1-self.boundaries_width)*self.pixel_size, c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
			for i in range (1,self.N_domains-1):
				plt.axvline(x=((i+1)*(self.n1-self.overlap)-self.boundaries_width)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
				plt.axvline(x=(i*(self.n1-self.overlap)+self.n1-self.boundaries_width)*self.pixel_size, c='b', ls='dashdot', lw=1.5)
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
		plt.plot(self.x, np.real(self.u_true-self.u)*10, 'g', lw=1., label='Error*10')
		plt.title('Field (Relative Error = {:.2e})'.format(self.rel_err))
		plt.legend(ncols=2, framealpha=0.6)

		plt.subplot(2,1,2)
		res_plots = plt.loglog(np.arange(1,self.iters+2, self.iter_step), self.residual, lw=1.5)
		if self.N_domains > 1:
			plt.legend(handles=iter(res_plots), labels=tuple(f'{i+1}' for i in range(self.N_domains)), title='Subdomains', ncols=int(self.N_domains/4)+1, framealpha=0.5)
		plt.axhline(y=self.threshold_residual, c='k', ls=':')
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
		self.plot_basics(plt)
		line, = plt.plot([] , [], 'r', lw=2., animated=True, label='AnySim')
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
	def pad_func(self, M, M_roi, which_end='Both'):
		# boundary_ = lambda x: (np.arange(1,x+1)-0.21).T/(x+0.66)
		boundary_ = lambda x: np.interp(np.arange(x), [0,x-1], [0.04981993,0.95018007])
		left_boundary = boundary_(np.floor(self.boundaries_width))
		right_boundary = boundary_(np.ceil(self.boundaries_width))
		if which_end == 'Both':
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
