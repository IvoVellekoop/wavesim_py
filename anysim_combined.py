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
try:
	plt.rcParams['text.usetex'] = True
except:
	pass
from PIL.Image import open, BILINEAR, fromarray

class AnySim():
	def __init__(self, topic='Helmholtz', test='FreeSpace', small_circ_prob=False, wrap_around='boundaries'):
		self.topic = topic	# Helmholtz (the only one for now) or Maxwell
		self.test = test 	# 'FreeSpace', '1D', '2D', OR '2D_low_contrast'
		self.small_circ_prob = small_circ_prob # True (V0 as in AnySim) or False (V0 as in WaveSim)
		self.wrap_around = wrap_around # 'boundaries', 'L_Omega', OR 'L_corr'

		self.max_iters = int(6.e+5)		# Maximum number of iterations
		self.iters = self.max_iters - 1
		self.lambd = 1.				# Wavelength in um (micron)
		self.N_roi = 128			# Num of points in ROI (Region of Interest)
		self.boundaries_width = 0	# Width of boundary layer to remove wrap-around effects. 0 in new approach with correction term

		''' Create log folder / Check for existing log folder'''
		today = date.today()
		d1 = today.strftime("%Y%m%d")
		self.log_dir = 'logs/Logs_'+d1+'/'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		self.run_id = d1 + '_' + self.test + '_scp' + str(int(self.small_circ_prob)) + '_wrap_' + self.wrap_around
		self.run_loc = self.log_dir + self.run_id
		if not os.path.exists(self.run_loc):
			os.makedirs(self.run_loc)

	def runit(self): 		# function that calls all the other 'main' functions
		s1 = time.time()
		self.init_setup()		# set up the grid, ROI, source, etc. based on the test
		self.print_details()	# print the simulation details
		self.make_medium()		# B = 1 - V
		self.make_propagator()	# L and (L+1)^(-1)

		# Scale the source term (and pad if boundaries)
		if self.wrap_around == 'boundaries':
			self.b = self.Tl * np.pad(self.b, self.N_dim*((self.bw_l,self.bw_r),), mode='constant') # source term y
		else:
			self.b = self.Tl * self.b
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.iterate()
		
		# Truncate u to ROI
		if self.wrap_around == 'boundaries':
			if self.N_dim == 1:
				self.u = self.u[self.bw_l:-self.bw_r]
				self.u_iter = self.u_iter[:, self.bw_l:-self.bw_r]
			elif self.N_dim == 2:
				self.u = self.u[self.bw_l:-self.bw_r,self.bw_l:-self.bw_r]
				self.u_iter = self.u_iter[:, self.bw_l:-self.bw_r,self.bw_l:-self.bw_r]

		print('Simulation done (Time {} s). Plotting...'.format(np.round(time.time()-s1,2)))
		self.plot_details()	# Plot the final field, residual vs. iterations, and animation of field with iterations
		return self.u

	def init_setup(self):
		if self.test == 'FreeSpace' or self.test == '1D':
			self.N_dim = 1
			self.pixel_size = self.lambd/4
			if self.test == 'FreeSpace':
				if self.wrap_around == 'boundaries':
					self.boundaries_width = 128  # matlab 1024
				self.n = np.ones(self.N_roi)
			elif self.test == '1D':
				self.N_roi = 256
				if self.wrap_around == 'boundaries':
					self.boundaries_width = 64
				self.n = np.ones(self.N_roi)
				self.n[99:130] = 1.5
			## define a point source
			source_amplitude = 1.
			self.b = np.zeros((self.N_roi,), dtype='complex_')
			self.b[0] = source_amplitude
		elif self.test == '2D' or self.test == '2D_low_contrast':
			self.N_dim = 2
			oversampling = 0.25
			im = np.asarray(open('../anysim_modified/tests/logo_structure_vector.png'))/255
			self.lambd = 0.532
			if self.test == '2D':
				n_iron = 2.8954 + 2.9179j
				n_contrast = n_iron - 1
				n_im = ((np.where(im[:,:,2]>(0.25),1,0) * n_contrast)+1)
				self.n = loadmat('../anysim_modified/n2d.mat')['n']
				# print(relative_error(n2d, self.n))
				if self.wrap_around == 'boundaries':
					self.boundaries_width = 31.5
				self.max_iters = int(1.e+2)  # 1.e+4 iterations gives relative error 1.65e-4 with the matlab test result, but takes ~140s
				self.pixel_size = self.lambd/(3*np.max(abs(n_contrast+1)))
			elif self.test == '2D_low_contrast':
				n_water = 1.33
				n_fat = 1.46
				n_im = (np.where(im[:,:,2]>(0.25),1,0) * (n_fat-n_water)) + n_water
				self.n = loadmat('../anysim_modified/n2d_lc.mat')['n']
				# print(relative_error(n2d_lc, self.n))
				if self.wrap_around == 'boundaries':
					self.boundaries_width = 75
				self.max_iters = 130
				self.pixel_size = self.lambd/(3*abs(n_fat))
			self.N_roi = int(oversampling*n_im.shape[0])
			# self.n = np.asarray(fromarray(n_im).resize((self.N_roi,self.N_roi), BILINEAR)) # In '2D' test case, resize cannot work with complex values?
			self.b = np.asarray(fromarray(im[:,:,1]).resize((self.N_roi,self.N_roi), BILINEAR))
		self.k0 = (1.*2.*np.pi)/(self.lambd)  # wavevector k = 2*pi/lambda, where lambda = 1.0 um (micron), and thus k0 = 2*pi = 6.28...
		self.N = int(self.N_roi+2*self.boundaries_width)
		self.bw_l = int(np.floor(self.boundaries_width))
		self.bw_r = int(np.ceil(self.boundaries_width))

		if self.test == 'FreeSpace':
			## Compare with analytic solution
			x = np.zeros((self.N_roi+2*128))
			x[np.where(x==0)]=np.nan
			x[128:-128] = np.arange(0,self.N_roi*self.pixel_size,self.pixel_size)
			h = self.pixel_size
			k = self.k0 * np.array([1.0])
			phi = k * x

			E_theory = 1.0j*h/(2*k) * np.exp(1.0j*phi) - h/(4*np.pi*k) * (np.exp(1.0j * phi) * ( np.exp(1.0j * (k-np.pi/h) * x) - np.exp(1.0j * (k+np.pi/h) * x)) - np.exp(-1.0j * phi) * ( -np.exp(-1.0j * (k-np.pi/h) * x) + np.exp(-1.0j * (k+np.pi/h) * x)))
			# special case for values close to 0
			small = np.abs(k*x) < 1.e-10
			E_theory[small] = 1.0j * h/(2*k) * (1 + 2j * np.arctanh(h*k/np.pi)/np.pi); # exact value at 0.
			self.E_theory = E_theory[128:-128]

	def print_details(self):
		# print('Topic: \t', self.topic)
		print('Test: \t', self.test)
		print('Smallest circle problem for V0?: \t', self.small_circ_prob)
		print('Tackling wrap-around effects with: \t', self.wrap_around)
		print('Boundaries width: ', self.boundaries_width)

	# Medium. B = 1 - V
	def make_medium(self):
		Vraw = self.k0**2 * self.n**2
		# give tiny non-zero minimum value to prevent division by zero in homogeneous media
		try:
			mu_min = 10.0/(self.boundaries_width * self.pixel_size)
		except:
			mu_min = 0
		mu_min = max( mu_min, 1.e+0/(self.N*self.pixel_size) )
		Vmin = np.imag( (self.k0 + 1j*np.max(mu_min))**2 )
		Vmax = 0.95
		if self.small_circ_prob == True:
			self.center_scale(-1j*Vraw, np.array([Vmin]), Vmax)
			self.V0 = 1j*self.V0
		else:
			self.V0 = (np.max(np.real(Vraw)) + np.min(np.real(Vraw)))/2 + 1j*Vmin
			self.V = -1j * (Vraw - self.V0)
		if self.wrap_around == 'L_corr':
			cp = 20
			F = self.DFT_matrix(self.N)
			Finv = np.asarray(np.matrix(F).H/self.N)
			p = 2*np.pi*np.fft.fftfreq(self.N, self.pixel_size)
			L = p**2
			Lw = Finv @ np.diag(L.flatten()) @ F
			L_corr = -np.real(Lw)                          # copy -Lw
			L_corr[:-cp,:-cp] = 0; L_corr[cp:,cp:] = 0  # Keep only upper and lower traingular corners of -Lw
			self.V = np.diag(self.V) + 1j * L_corr
		scaling = Vmax/self.checkV(self.V)
		self.V = scaling * self.V

		self.Tr = np.sqrt(scaling)
		self.Tl = 1j * self.Tr

		## Check that ||V|| < 1 (0.95 here)
		vc = self.checkV(self.V)
		if vc < 1:
			pass
		else:
			return print('||V|| not < 1, but {}'.format(vc))

		## B = 1 - V
		if self.wrap_around == 'L_corr':
			B = np.eye(self.N) - self.V
			self.medium = lambda x: B @ x
		elif self.wrap_around == 'L_Omega':
			B = 1 - self.V
			self.medium = lambda x: B * x
		else:
			B = self.pad_func((1-self.V)).astype('complex_')
			self.medium = lambda x: B * x	

	# Propagator. (L+1)^(-1)
	def make_propagator(self):
		if self.wrap_around == 'L_Omega':
			N = self.N*10
		else:
			N = self.N

		L = (self.coordinates_f(1, N)**2).T
		for d in range(2,self.N_dim+1):
			L = L + self.coordinates_f(d, N)**2
		L = self.Tl * self.Tr * (L - self.V0)
		Lr = np.squeeze(1/(L+1))
		if self.wrap_around == 'L_Omega':
			self.propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn( np.pad(x,(0,N-self.N)) )))[:self.N]
		else:
			self.propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

		## Check that A = L + V is accretive
		if self.N_dim == 1:
			if self.wrap_around == 'boundaries':
				L = np.diag(np.squeeze(L)[self.bw_l:-self.bw_r])
				self.V = np.diag(self.V)
			elif self.wrap_around == 'L_corr':
				L = np.diag(np.squeeze(L))
			
			if self.wrap_around == 'boundaries' or self.wrap_around == 'L_corr':
				A = L + self.V
				acc = np.min(np.linalg.eigvals(A + np.asarray(np.matrix(A).H)))
				if np.round(acc, 13) < 0:
					return print('A is not accretive. ', acc)

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
			residual_i = nr/normb
			residual.append(residual_i)
			if residual_i < 1.e-6:
				self.iters = i
				print('Stopping simulation at iter {}, residual {:.2e} <= 1.e-6'.format(self.iters+1, residual_i))
				break
			self.u = self.u - (self.alpha * t1)
			u_iter.append(self.u)
		self.u = self.Tr * self.u

		self.u_iter = self.Tr.flatten() * np.array(u_iter)
		self.residual = np.array(residual)

	# Plotting functions
	def plot_details(self):
		self.plot_FieldNResidual()	# png
		# self.plot_field_final()	# png
		# self.plot_residual()		# png
		self.plot_field_iters()		# movie/animation/GIF
		print('Plotting done.')

	def plot_FieldNResidual(self): # png
		plt.subplots(figsize=figsize, ncols=1, nrows=2)

		plt.subplot(2,1,1)
		x = np.arange(self.N_roi)*self.pixel_size
		if self.test == 'FreeSpace':
			u_true = self.E_theory.copy()
			plt.plot(x, np.real(u_true), 'k', lw=2., label='Analytic solution')
		elif self.test == '1D':
			u_true = np.squeeze(loadmat('anysim_matlab/u.mat')['u'])
			plt.plot(x, np.real(u_true), 'k', lw=2., label='Matlab solution')
		plt.plot(x, np.real(self.u), 'r', lw=1., label='RelErr = {:.2e}'.format(self.relative_error(self.u,u_true)))
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

		if self.wrap_around == 'boundaries':
			plt.suptitle(f'Tackling wrap-around effects with: {self.wrap_around} (width = {int(self.boundaries_width)})')
		else:
			plt.suptitle(f'Tackling wrap-around effects with: {self.wrap_around}')
		plt.tight_layout()
		plt.savefig(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual.png', bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.draw()

	def plot_field_iters(self): # movie/animation/GIF
		self.u_iter = np.real(self.u_iter)
		x = np.arange(self.N_roi)*self.pixel_size

		fig = plt.figure(figsize=(14.32,8))
		if self.test == 'FreeSpace':
			plt.plot(x, np.real(self.E_theory), 'k:', lw=0.75, label='Analytic solution')
		elif self.test == '1D':
			plt.plot(x, np.real(np.squeeze(loadmat('anysim_matlab/u.mat')['u'])), 'k:', lw=0.75, label='Matlab solution')
		plt.xlabel("$x$")
		plt.ylabel("Amplitude")
		plt.xlim([x[0]-x[1]*2,x[-1]+x[1]*2])
		plt.ylim([np.min(self.u_iter), np.max(self.u_iter)])
		plt.grid()
		line, = plt.plot([] , [], 'b', lw=1., animated=True)
		line.set_xdata(x)
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
			if self.wrap_around == 'boundaries':
				title.set_text(f'Tackling wrap-around effects with: {self.wrap_around} (width = {int(self.boundaries_width)}). Iteration {iters_trunc[i]+1}')
			else:
				title.set_text(f'Tackling wrap-around effects with: {self.wrap_around}. Iteration {iters_trunc[i]+1}')
			return line, title,
		ani = animation.FuncAnimation(
			fig, animate, interval=100, blit=True, frames=plot_iters)
		writer = animation.FFMpegWriter(
		    fps=10, metadata=dict(artist='Me'))
		ani.save(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_Field.mp4', writer=writer)
		plt.close()

	## smallest circle problem
	def center_scale(self, Vraw, Vmin, Vmax):
		dim = sum(Vmin.shape[a-1]>1 for a in range(Vmin.ndim))
		Vreshape = np.expand_dims(Vraw, axis=tuple(range(2-dim)))
		Vmin = np.expand_dims(Vmin, axis=tuple(range(2-Vmin.ndim)))
		N = Vreshape.shape[0]
		M = Vreshape.shape[1]
		centers = np.zeros((N, M), dtype='complex_')
		radii = np.zeros((N, M))

		for n in range(N):
			for m in range(M):
				c, r = self.smallest_circle((Vreshape[n,m,:]))
				# adjust centers and radii so that the real part of centers+radii >= Vmin
				re_diff = np.real(c) + r - Vmin[n, m]
				if re_diff < 0:
					c = c - re_diff/2
					r = r - re_diff/2
					print('slowing down simulation to accomodate boundary conditions')
				centers[n,m] = c
				radii[n,m] = r
		
		if dim == 0: # potential is a scalar field
			if Vraw.any() < 0:
				print('Vraw is not accretive')
			else:
				Ttot = Vmax/radii
				Tl = np.sqrt(Ttot).flatten()
				Tr  = Tl.copy()
				self.V0 = centers.copy()
				# self.V = Ttot.flatten() * (Vraw - self.V0)
				self.V = Vraw - self.V0.flatten()
		elif dim == 1: # potential is a field of diagonal matrices, stored as column vectors
			if Vraw.any() < 0:
				print('Vraw is not accretive')
			else:
				if (radii < np.abs(centers) * 1.e-6).any():
					radii = np.maximum(radii, np.abs(c)*1.e-6)
					print('At least one of the components of the potential is (near-)constant, using threshold to avoid divergence in Tr')
				TT = Vmax/radii.flatten()
				Tl = np.sqrt(np.diag(TT)).flatten()
				Tr = Tl.copy()
				self.V0 = np.diag(centers.flatten())
				# self.V = TT * (Vraw - np.diag(self.V0))
				self.V = Vraw - np.diag(self.V0)
		# elif dim == 2: # potential is a field of full matrices, stored as pages
		#     # check if matrix is near-singular
		#     # and add a small offset to the singular values if it is
		#     U, S, Vh = np.linalg.svd(radii)
		#     V = Vh.T
		#     cS = np.diag(U.T @ centers @ V)
		#     if (np.diag(S) < np.abs(cS) * 1.e-6).any():
		#         S = np.maximum(S, np.abs(cS)*1.e-6)
		#         radii = U @ S @ V
		#         ('At least one of the components of the potential is (near-)constant, using threshold to avoid divergence in Tr')
		#     # compute scaling factor for the matrix
		#     P, R, C = equilibriate(radii) ### python equivalent function? scipy.linalg.matrix_balance?

	def smallest_circle(self, points, tolerance=1.e-10):
		points = points.flatten()
		if np.isreal(points).all():
			pmin = np.min(points)
			pmax = np.max(points)
			center = (pmin + pmax)/2
			radius = pmax - center
			return center, radius

		N_reads = 0

		# Step 0, pick four initial corner points based on bounding box
		corner_i = np.zeros(4, dtype=int)
		corner_i[0] = np.argmin(np.real(points))
		corner_i[1] = np.argmax(np.real(points))
		corner_i[2] = np.argmin(np.imag(points))
		corner_i[3] = np.argmax(np.imag(points))
		p_o = np.zeros_like(corner_i, dtype='complex')
		for a in range(corner_i.size):
			p_o[a] = points[corner_i[a]]
		width = np.real(p_o[1] - p_o[0])
		height = np.imag(p_o[3] - p_o[2])
		r_o = (np.sqrt(width**2 + height**2) / 2).astype('complex_')
		center = 0.5 * (np.real(p_o[0] + p_o[1]) + 1j * np.imag(p_o[2] + p_o[3]))
		N_reads = N_reads + points.size

		for _ in range(50):
			# step 1
			'''
			here, p_o contains up to 7 points, pick the 2-3 that correspond
			to the smallest circle enclosing them all
			sort in order of increasing distance from center since it is
			more likely that the final circle will be built from the points
			further away.
			'''
			ind = np.argsort(np.abs(p_o - center))
			center, radius, p_o = self.smallest_circle_brute_force(np.array([p_o[a] for a in ind]))

			# step 2
			c_c = self.conjugate_inflated_triangle(p_o, r_o)

			# 2a: select points
			try:
				distances = np.abs(np.expand_dims(points, axis=1) - np.expand_dims(np.concatenate(([center], c_c)), axis=0))
			except:
				distances = np.abs(np.expand_dims(points, axis=1) - np.expand_dims(np.concatenate((center, c_c)), axis=0))

			# print('distances', distances.shape, distances)
			keep = np.max(distances, 1) > r_o# - tolerance
			N_reads = N_reads + keep.size
			
			# 2b: determine outliers
			r_out = np.max(distances, 0)
			outliers_i = np.argmax(distances, 0)
			if r_out.flatten()[0] < radius + tolerance:
				radius = r_out.flatten()[0]
				return center, radius
			outliers = points[outliers_i]
			r_o = np.minimum(np.min(r_out), r_o)
			points = np.concatenate((points[keep], p_o))
			p_o = np.concatenate((outliers, p_o))

		return center, radius

	def smallest_circle_brute_force(self, points, tolerance=1.e-10):
		N = len(points)
		if N==1:
			center = points.copy()
			radius = 0
			corners = points.copy()
			return center, radius, corners
		elif N==2:
			center = (points[0] + points[1])/2
			radius = np.abs(points[0]-center)
			corners = points.copy()
			return center, radius, corners
			
		'''
		Remove one point and recursively construct a smallest circle for the
		remaining points. If the removed point is inside that circle, return
		if the removed point is not in the circle, repeat with a different point
		omitted. First check if it is possible to construct a circle from just 2 points,
		including the third
		todo: faster check to see if two or three points are needed?
		'''
		Ns = np.arange(N)
		for p in range(N):
			reduced = points[[Ns[a]!=p for a in range(len(Ns))]].copy()
			center, radius, corners = self.smallest_circle_brute_force(reduced, tolerance=1.e-10)
			if np.abs(points[p]-center) <= radius + tolerance:
				return center, radius, corners
			
		# if we get here, no suitable subset was found. This is only possible for 3 points

		## All three points are edge points
		# now write in matrix form and let Python (MATLAB) solve it
		A = points[0]
		B = points[1]
		C = points[2]
		M = 2 * np.array([[np.real(A)-np.real(B), np.imag(A)-np.imag(B)],[np.real(A)-np.real(C), np.imag(A)-np.imag(C)]])
		b = np.array([[np.abs(A)**2 - np.abs(B)**2],[np.abs(A)**2 - np.abs(C)**2]])

		try:
			c = np.linalg.solve(M, b)
		except:
			c= np.linalg.lstsq(M, b)
		center = c[0] + 1j * c[1]
		radius = np.abs(A-center)
		corners = points

		return center, radius, corners

	def conjugate_inflated_triangle(self, points, r):
		c_c = np.zeros(3, dtype='complex_')
		Np = len(points)
		if Np==2:
			B = points[0]
			C = points[1]
			M_start = (C+B)/2
			M_dir = 1j * (C-B)/np.abs(C-B)
			w = np.abs(C - M_start)
			alpha = np.sqrt(r**2 - w**2)
			c_c[0] = M_start - alpha*M_dir
			c_c[1] = M_start + alpha*M_dir

			# not needed, but we can pick one point 'for free'
			M_dir = (C-B) / np.abs(C-B)
			c_c[2] = B + M_dir*r
			return c_c
		
		ss = self.signed_surface(points)
		if ss < 0: # reverse direction of circle (otherwise we get the solutions outside of the circle)
			tmp = points[0]
			points[0] = points[2]
			points[2] = tmp

		for p in range(Np):
			'''
			For conjugating point A, define mid-point M_CB and line M_CB-A
			c_A is on this line, at a distance of r from point B (and point
			C)
			%
			c_A = M_CB + alpha * (A - M_CB) / |A-M_CB|  with alpha >= 0 
			w = |C - M_CB|
			h = alpha
			w^2 + alpha^2 = r^2
			alpha = sqrt(r^2 - w^2)
			'''
			B = points[p]
			C = points[np.remainder(p+1, Np)]
			M_start = (C+B)/2
			M_dir = 1j * (C-B)/np.abs(C-B)
			w = np.abs(C - M_start)
			alpha = np.sqrt(r**2 - w**2)
			c_c[p] = M_start + alpha*M_dir

		return c_c

	def signed_surface(self, points):
		return np.imag((points[0]-points[1]) * np.conj(points[2]-points[1]))

	## pad boundaries
	def pad_func(self, M, element_dimension=0):
		sz = M.shape[element_dimension+0:self.N_dim]
		if (self.boundaries_width != 0) & (sz == (1,)):
			M = np.tile(M, (int((np.ones(1) * (self.N_roi-1) + 1)[0]), 1))

		M = np.pad(M, ((self.bw_l,self.bw_r)), mode='edge')

		for d in range(self.N_dim):
			try:
				w = self.boundaries_width[d]
			except:
				w = self.boundaries_width

		if w>0:
			left_boundary = self.boundaries_window(np.floor(w))
			right_boundary = self.boundaries_window(np.ceil(w))
			full_filter = np.concatenate((left_boundary, np.ones((self.N_roi,1)), np.flip(right_boundary)))
			if self.N_dim == 1:
				M = M * np.squeeze(full_filter)
			else:
				M = full_filter.T * M * full_filter
		return M

	def boundaries_window(self, L):
		x = np.expand_dims(np.arange(L)/(L-1), axis=1)
		a2 = np.expand_dims(np.array([-0.4891775, 0.1365995/2, -0.0106411/3]) / (0.3635819 * 2 * np.pi), axis=1)
		return np.sin(x * np.expand_dims(np.array([1, 2, 3]), axis=0) * 2 * np.pi) @ a2 + x

	## DFT matrix
	def DFT_matrix(self, N):
		l, m = np.meshgrid(np.arange(N), np.arange(N))
		omega = np.exp( - 2 * np.pi * 1j / N )
		return np.power( omega, l * m )

	## Fourier co-ordinates
	def coordinates_f(self, dimension, N):
		pixel_size_f = 2 * np.pi/(self.pixel_size*N)
		k = np.expand_dims( self.fft_range(N) * pixel_size_f, axis=tuple(range(2-dimension)))
		return k

	def fft_range(self, N):
		return np.fft.ifftshift(self.symrange(N))

	def symrange(self, N):
		return range(-int(np.floor(N/2)),int(np.ceil(N/2)))

	## Relative error
	def relative_error(self, E_, E_true):
		return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )

	## Spectral radius
	def checkV(self, A):
		if self.wrap_around == 'L_corr':
			return np.linalg.norm(A,2)
		else:
			return np.max(np.abs(A))
