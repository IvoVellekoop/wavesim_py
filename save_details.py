from anysim_main import AnySim

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
font = {'family':'Times New Roman', # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size':18}                      # 8-10 pt
rc('font',**font)
figsize = (8,8) #(14.32,8)

class LogPlot(AnySim):
	def __init__(self):
		self.log_details()
		self.plot_details()
		self.plotting_done = True

	# Save some parameters and stats
	def log_details(self):
		print('Saving stats...')
		save_string = f'N_dims {AnySim.N_dims}; boundaries width {AnySim.boundary_widths}; N_domains {AnySim.N_domains}; overlap {AnySim.overlap}'
		if AnySim.wrap_correction:
			save_string += f'; wrap correction {AnySim.wrap_correction}; corner points {AnySim.cp}'
		save_string += f'; {AnySim.sim_time:>2.2f} sec; {AnySim.iters+1} iterations; final residual {AnySim.residual_i:>2.2e}'
		if 'AnySim.rel_err' in globals():
			save_string += f'; relative error {AnySim.rel_err:>2.2e}'
		save_string += f' \n'
		with open(AnySim.stats_file_name,'a') as fileopen:
			fileopen.write(save_string)

	# Plotting functions
	def plot_details(self):
		print('Plotting...')

		if 'AnySim.u_true' in globals():
			self.label = 'Reference solution'

		if AnySim.N_dims == 1:
			self.x = np.arange(AnySim.N_roi[0])*AnySim.pixel_size
			self.plot_FieldNResidual()	# png
			# AnySim.plot_field_iters()		# movie/animation/GIF
		elif AnySim.N_dims == 2:
			self.image_FieldNResidual()	# png
		elif AnySim.N_dims == 3:
			for z_slice in [0, int(AnySim.u.shape[2]/2), int(AnySim.u.shape[2]-1)]:
				self.image_FieldNResidual(z_slice)	# png
		plt.close('all')
		print('Plotting done.')

	def plot_basics(self, plt):
		if AnySim.total_domains > 1:
			plt.axvline(x=(AnySim.domain_size[0]-AnySim.boundary_widths[0]-AnySim.overlap[0])*AnySim.pixel_size, c='b', ls='dashdot', lw=1.5)
			plt.axvline(x=(AnySim.domain_size[0]-AnySim.boundary_widths[0])*AnySim.pixel_size, c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
			for i in range (1,AnySim.total_domains-1):
				plt.axvline(x=((i+1)*(AnySim.domain_size[0]-AnySim.overlap[0])-AnySim.boundary_widths[0])*AnySim.pixel_size, c='b', ls='dashdot', lw=1.5)
				plt.axvline(x=(i*(AnySim.domain_size[0]-AnySim.overlap[0])+AnySim.domain_size[0]-AnySim.boundary_widths[0])*AnySim.pixel_size, c='b', ls='dashdot', lw=1.5)
		if 'AnySim.u_true' in globals():
			plt.plot(self.x, np.real(AnySim.u_true), 'k', lw=2., label=self.label)
		plt.ylabel('Amplitude')
		plt.xlabel("$x~[\lambda]$")
		plt.xlim([self.x[0]-self.x[1]*2,self.x[-1]+self.x[1]*2])
		plt.grid()

	def plot_FieldNResidual(self): # png
		plt.subplots(figsize=figsize, ncols=1, nrows=2)

		plt.subplot(2,1,1)
		self.plot_basics(plt)
		plt.plot(self.x, np.real(AnySim.u), 'r', lw=1., label='AnySim')
		title = 'Field'
		if 'AnySim.u_true' in globals():
			plt.plot(self.x, np.real(AnySim.u_true-AnySim.u)*10, 'g', lw=1., label='Error*10')
			title += f' (Relative Error = {AnySim.rel_err:.2e})'
		plt.title(title)
		plt.legend(ncols=2, framealpha=0.6)

		plt.subplot(2,1,2)
		res_plots = plt.loglog(np.arange(1,AnySim.iters+2, AnySim.iter_step), AnySim.residual, lw=1.5)
		if AnySim.total_domains > 1:
			plt.legend(handles=iter(res_plots), labels=tuple(f'{i+1}' for i in AnySim.range_total_domains), title='Subdomains', ncols=int(AnySim.N_domains[0]/4)+1, framealpha=0.5)
		plt.loglog(np.arange(1,AnySim.iters+2, AnySim.iter_step), AnySim.full_residual, lw=3., c='k', ls='dashed', label='Full Residual')
		plt.axhline(y=AnySim.threshold_residual, c='k', ls=':')
		plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
		ymin = np.minimum(6.e-7, 0.8*np.nanmin(AnySim.residual))
		ymax = np.maximum(2.e+0, 1.2*np.nanmax(AnySim.residual))
		plt.ylim([ymin, ymax])
		plt.title('Residual. Iterations = {:.2e}'.format(AnySim.iters+1))
		plt.ylabel('Residual')
		plt.xlabel('Iterations')
		plt.grid()

		title_text = ''
		title_text = f'{title_text} Absorbing boundaries ({AnySim.boundary_widths}). '
		if AnySim.wrap_correction:
			title_text += f'{title_text} Wrap correction: {AnySim.wrap_correction}. '
		plt.suptitle(title_text)

		plt.tight_layout()
		fig_name = f'{AnySim.run_loc}/{AnySim.run_id}_{AnySim.iters+1}iters_FieldNResidual'
		if AnySim.wrap_correction == 'L_corr':
			fig_name += f'_cp{AnySim.cp}'
		fig_name += f'.png'
		plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.close('all')

	def plot_field_iters(self): # movie/animation/GIF
		# AnySim.u_iter = np.real(AnySim.u_iter)

		fig = plt.figure(figsize=(14.32,8))
		AnySim.plot_basics(plt)
		line, = plt.plot([] , [], 'r', lw=2., animated=True, label='AnySim')
		line.set_xdata(self.x)
		title = plt.title('')
		plt.legend()

		# Plot 100 or fewer frames. Takes much longer for any more frames.
		if AnySim.iters > 100:
			plot_iters = 100
			iters_trunc = np.linspace(0,AnySim.iters-1,plot_iters).astype(int)
			# u_iter_trunc = AnySim.u_iter[iters_trunc]
		else:
			plot_iters = AnySim.iters
			iters_trunc = np.arange(AnySim.iters)
			# u_iter_trunc = AnySim.u_iter

		def animate(i):
			# line.set_ydata(u_iter_trunc[i])		# update the data.
			title_text = f'Iteration {iters_trunc[i]+1}. '
			title_text = f'{title_text} Absorbing boundaries ({AnySim.boundary_widths}). '
			if AnySim.wrap_correction:
				title_text += f'{title_text} Wrap correction: {AnySim.wrap_correction}. '
			title.set_text(title_text)
			return line, title,
		ani = animation.FuncAnimation(
			fig, animate, interval=100, blit=True, frames=plot_iters)
		writer = animation.FFMpegWriter(
		    fps=10, metadata=dict(artist='Me'))
		ani_name = f'{AnySim.run_loc}/{AnySim.run_id}_{AnySim.iters+1}iters_Field'
		if AnySim.wrap_correction == 'L_corr':
			ani_name += f'_cp{AnySim.cp}'
		ani_name += f'.mp4'
		ani.save(ani_name, writer=writer)
		plt.close('all')

	def image_FieldNResidual(self, z_slice=0): # png
		if AnySim.N_dims == 3:
			u = AnySim.u[:,:,z_slice]
			if 'AnySim.u_true' in globals():
				u_true = AnySim.u_true[:,:,z_slice]
		else:
			u = AnySim.u.copy()
			if 'AnySim.u_true' in globals():
				u_true = AnySim.u_true.copy()

		if 'AnySim.u_true' in globals():
			nrows = 2
			vlim = np.maximum(np.max(np.abs(np.real(u_true))), np.max(np.abs(np.real(u_true))))
		else:
			nrows = 1
			vlim = np.maximum(np.max(np.abs(np.real(u))), np.max(np.abs(np.real(u))))
		plt.subplots(figsize=figsize, ncols=2, nrows=nrows)
		pad = 0.03; shrink = 0.65# 1.# 


		plt.subplot(2,2,1)
		plt.imshow(np.real(u), cmap='seismic', vmin=-vlim, vmax=vlim)
		plt.colorbar(shrink=shrink, pad=pad)
		plt.title('AnySim')

		plt.subplot(2,2,2)
		plt.loglog(np.arange(1,AnySim.iters+2, AnySim.iter_step), AnySim.full_residual, lw=3., c='k', ls='dashed')
		plt.axhline(y=AnySim.threshold_residual, c='k', ls=':')
		plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
		ymin = np.minimum(6.e-7, 0.8*np.nanmin(AnySim.residual))
		ymax = np.maximum(2.e+0, 1.2*np.nanmax(AnySim.residual))
		plt.ylim([ymin, ymax])
		plt.title('Residual. Iterations = {:.2e}'.format(AnySim.iters+1))
		plt.ylabel('Residual')
		plt.xlabel('Iterations')
		plt.grid()

		if 'AnySim.u_true' in globals():
			plt.subplot(2,2,3)
			plt.imshow(np.real(u_true), cmap='seismic', vmin=-vlim, vmax=vlim)
			plt.colorbar(shrink=shrink, pad=pad)
			plt.title(self.label)

			plt.subplot(2,2,4)
			plt.imshow(np.real(u_true-u), cmap='seismic')#, vmin=-vlim, vmax=vlim)
			plt.colorbar(shrink=shrink, pad=pad)
			plt.title(f'Difference. Relative error {AnySim.rel_err:.2e}')

		plt.tight_layout()

		title_text = ''
		title_text = f'{title_text} Absorbing boundaries ({AnySim.boundary_widths}). '
		if AnySim.wrap_correction:
			title_text += f'{title_text} Wrap correction: {AnySim.wrap_correction}. '
		plt.suptitle(title_text)

		plt.tight_layout()
		fig_name = f'{AnySim.run_loc}/{AnySim.run_id}_{AnySim.iters+1}iters_FieldNResidual_{z_slice}'
		if AnySim.wrap_correction == 'L_corr':
			fig_name += f'_cp{AnySim.cp}'
		fig_name += f'.png'
		plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
		plt.close('all')

