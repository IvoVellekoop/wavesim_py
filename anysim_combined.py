import os, time
import numpy as np
from datetime import date
import scipy
from scipy.linalg import eigvals
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (8,8) #(14.32,8)
from PIL.Image import open, BILINEAR, fromarray

class AnySim():
	def __init__(self, wrap_correction=False, absorbing_boundaries=True):
		self.wrap_correction = wrap_correction				# True to use the new wrap-around correction
		self.absorbing_boundaries = absorbing_boundaries	# True to add absorbing boundaries
		self.max_iters = int(1.e+7)		# Maximum number of iterations
		self.iters = self.max_iters - 1
		self.lambd = 1.					# Wavelength in um (micron)
		self.k0 = 2.*np.pi/self.lambd  	# Wavevector k0=2*pi/lambda, where lambda=1.0 um (micron). Thus k0=2*pi=6.28...
		self.pixel_size = self.lambd/4
		self.N_roi = 128				# Num of points in ROI (Region of Interest)
		if absorbing_boundaries:
			self.boundaries_width = 128
		else:
			self.boundaries_width = 0

		# set up coordinate ranges
		self.N = self.N_roi+2*self.boundaries_width
		self.x = np.arange(-self.boundaries_width, self.N - self.boundaries_width) * self.pixel_size
		self.p = 2 * np.pi * np.fft.fftfreq(self.N, self.pixel_size) 

		''' Create log folder / Check for existing log folder'''
		today = date.today()
		d1 = today.strftime("%Y%m%d")
		self.log_dir = 'logs/Logs_'+d1+'/'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)

		if self.absorbing_boundaries:
			d1 = d1 + '_abs'
		if self.wrap_correction:
			d1 = d1 + '_corr'
		
		self.run_id = d1
		self.run_loc = self.log_dir + self.run_id
		if not os.path.exists(self.run_loc):
			os.makedirs(self.run_loc)

	def runit(self):			# function that calls all the other 'main' functions
		s1 = time.time()
		self.init_setup()		# set up the grid, ROI, source, etc. based on the test
		self.print_details()	# print the simulation details
		self.make_operators()	# B = 1 - V and (L+1)^(-1)

		# Scale the source term (and pad if boundaries)
		self.b = self.scaling * np.pad(self.b, (self.boundaries_width, self.boundaries_width), mode='constant') # source term y
		self.u = (np.zeros_like(self.b, dtype='complex_'))	# field u, initialize with 0

		# AnySim update
		self.iterate()

		# Truncate u to ROI
		if self.absorbing_boundaries:
			self.u = self.u[self.boundaries_width:-self.boundaries_width]
			self.u_iter = self.u_iter[:, self.boundaries_width:-self.boundaries_width]

		# Print relative error between u and analytic solution
		self.rel_err = self.relative_error(self.u, self.E_theory)
		print('Relative error: {:.2e}'.format(self.rel_err))

		print('Simulation done (Time {} s). Plotting...'.format(np.round(time.time()-s1,2)))
		self.plot_details()	# Plot the final field, residual vs. iterations, and animation of field with iterations
		return self.u

	def init_setup(self):
		# set up a free space simulation. Note: constructing the refractive index should not happen in the AnySim object at all
		self.n = np.ones(self.N_roi)

		## define a point source with amplitude 1
		self.b = np.zeros(self.N_roi, dtype='complex_')
		self.b[0] = 1
		
		## compute analytic solution for free space propagation
		h = self.pixel_size
		k = self.k0
		if self.boundaries_width > 0:
			x = self.x[self.boundaries_width:-self.boundaries_width]
		else:
			x = self.x
		phi = k * x
		E_theory = 1.0j*h/(2*k) * np.exp(1.0j*phi) - h/(4*np.pi*k) * (np.exp(1.0j * phi) * ( np.exp(1.0j * (k-np.pi/h) * x) - np.exp(1.0j * (k+np.pi/h) * x)) - np.exp(-1.0j * phi) * ( -np.exp(-1.0j * (k-np.pi/h) * x) + np.exp(-1.0j * (k+np.pi/h) * x)))
		# special case for values close to 0
		small = np.abs(x) < 1.e-10
		E_theory[small] = 1.0j * h/(2*k) * (1 + 2j * np.arctanh(h*k/np.pi)/np.pi); # exact value at 0.
		self.E_theory = E_theory

	def print_details(self):
		print('Test: \t', self.run_id)
		print('Boundaries width: ', self.boundaries_width)

	# Medium. B = 1 - V
	def make_operators(self):
		# Vraw is the centered, non-scaled, non-rotated scattering potential 
		# Multiply by self.scale to get the canonical V
		Vrr = self.k0**2 * self.n**2
		self.V0 = (np.max(np.real(Vrr)) + np.min(np.real(Vrr)))/2
		Vrr = np.pad(Vrr, (self.boundaries_width, self.boundaries_width), mode='edge')
		
		Vraw = np.diag(Vrr - self.V0)	# always represent V as a matrix (not very efficient)

		# Lraw is the shifted, non-scaled, non-rotated laplacian.
		# Multiply by self.scale to get the canonical L
		# Lraw is represented as a vector. To apply it, point-wise multiply in the Fourier domain
		Lraw = self.V0 - self.p**2

		# Compute the (non-scaled) wrapping correction term
		if self.wrap_correction:
			cp = 20
			F = np.array(scipy.linalg.dft(self.N))
			Finv = np.conj(F).T / self.N
			L_corr = Finv @ np.diag(Lraw) @ F			# circular convolution matrix (with wrapping artefacts)
			L_corr[:-cp,:-cp] = 0; L_corr[cp:,cp:] = 0  # Keep only the wrapping artefacts
			Vraw = Vraw - L_corr						# subtract the wrapping artefacts

		# compute and apply scaling and shifting. Use a small minimum to avoid division by zero if the medium is completely homogeneous 
		self.scaling = 0.95j / max(np.linalg.norm(Vraw, 2), 1)
		self.V = -self.scaling * Vraw
		self.L = -self.scaling * Lraw

		# assert accretivity
		print(np.min(np.real(self.L)))
		print(np.min(np.real(np.linalg.eigvals(self.V + self.V.conj().T))))

		# apply absorbing boundaries (scales diagonal of B)
		B = np.eye(self.N) - self.V
		np.fill_diagonal(B, self.abs_func(B.diagonal().copy()))
		self.medium = lambda x: B @ x

		Lr = 1 / (self.L + 1)
		self.propagator = lambda x: (np.fft.ifftn(Lr * np.fft.fftn(x)))

	# AnySim update
	def iterate(self):
		self.alpha = 0.75									# ~step size of the Richardson iteration \in (0,1]
		residual = []
		u_iter = []	# for debugging, store all fields!
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

		self.u_iter = np.array(u_iter)
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
		plt.plot(x, np.real(self.E_theory), 'k', lw=2., label='Analytic solution')
		plt.plot(x, np.real(self.u), 'r', lw=1., label='RelErr = {:.2e}'.format(self.rel_err))
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
		plt.suptitle(self.run_id)
		plt.tight_layout()
		plt.savefig(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_FieldNResidual.png', bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.draw()

	def plot_field_iters(self): # movie/animation/GIF
		self.u_iter = np.real(self.u_iter)
		x = np.arange(self.N_roi)*self.pixel_size

		fig = plt.figure(figsize=(14.32,8))
		plt.plot(x, np.real(self.E_theory), 'k:', lw=0.75, label='Analytic solution')
		plt.xlabel("$x$")
		plt.ylabel("Amplitude")
		plt.xlim([x[0]-x[1]*2,x[-1]+x[1]*2])
		plt.ylim([1.05*np.min(self.u_iter), 1.05*np.max(self.u_iter)])
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
			# title.set_text(self.run_id + ":" + str(i))
			title.set_text(f'{self.run_id} : {iters_trunc[i]+1}')
			return line, title,
		ani = animation.FuncAnimation(
			fig, animate, interval=100, blit=True, frames=plot_iters)
		writer = animation.FFMpegWriter(
		    fps=10, metadata=dict(artist='Me'))
		ani.save(f'{self.run_loc}/{self.run_id}_{self.iters+1}iters_Field.mp4', writer=writer)
		# plt.show()
		plt.close()

	
	## absorbing boundaries
	def abs_func(self, M):
		left_boundary = self.boundaries_window(self.boundaries_width)
		right_boundary = self.boundaries_window(self.boundaries_width)
		full_filter = np.concatenate((left_boundary, np.ones(self.N_roi, ), np.flip(right_boundary)))
		M = M * full_filter
		return M

	def boundaries_window(self, L):
		x = np.expand_dims(np.arange(L)/(L-1), axis=1)
		a2 = np.expand_dims(np.array([-0.4891775, 0.1365995/2, -0.0106411/3]) / (0.3635819 * 2 * np.pi), axis=1)
		return np.squeeze(np.sin(x * np.expand_dims(np.array([1, 2, 3]), axis=0) * 2 * np.pi) @ a2 + x)

	## Relative error
	def relative_error(self, E_, E_true):
		return np.mean( np.abs(E_-E_true)**2 ) / np.mean( np.abs(E_true)**2 )

	
sim = AnySim(wrap_correction=False, absorbing_boundaries=True)
sim.runit()