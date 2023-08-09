from helmholtzbase import HelmholtzBase
from state import State

import os
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

font = {'family': 'Times New Roman',  # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size': 18}  # 8-10 pt
rc('font', **font)
figsize = (8, 8)  # (14.32,8)


def print_details(base: HelmholtzBase):
    print(f'\n{base.n_dims} dimensional problem')
    if base.wrap_correction:
        print('Wrap correction: \t', base.wrap_correction)
    print('Boundaries width: \t', base.boundary_widths)
    if base.total_domains > 1:
        print(
            f'Decomposing into {base.n_domains} domains of size {base.domain_size}, overlap {base.overlap}')


def relative_error(e, e_true):
    """ Relative error """
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)


class LogPlot:
    def __init__(self, base: HelmholtzBase, state: State, u_computed: np.array([]), u_reference: np.array([])):
        """ Logging and Plotting Class """
        self.base = base
        self.state = state
        self.u_computed = u_computed
        self.u_reference = u_reference
        self.rel_err = None
        self.plotting_done = None
        self.x = None

        # Create log folder / Check for existing log folder
        today = date.today()
        d1 = today.strftime("%Y%m%d")
        self.log_dir = 'logs/Logs_' + d1 + '/'
        if not os.path.exists(self.log_dir):
            print(f'"{self.log_dir}" does not exist, creating...')
            os.makedirs(self.log_dir)
        else:
            print(f'"{self.log_dir}" exists.')

        self.run_id = d1 + '_n_dims' + str(self.base.n_dims)
        self.run_loc = self.log_dir + self.run_id
        if not os.path.exists(self.run_loc):
            print(f'"{self.run_loc}" does not exist, creating...')
            os.makedirs(self.run_loc)
        else:
            print(f'"{self.run_loc}" exists.')

        self.run_id += '_abs' + str(self.base.boundary_widths)
        if self.base.wrap_correction:
            self.run_id += '_' + self.base.wrap_correction
        self.run_id += '_n_domains' + str(self.base.n_domains)

        self.stats_file_name = self.run_loc + '_stats.txt'

        if 'u_reference' in locals():
            self.label = 'Reference solution'

    def compare(self):
        """ Compute and print relative error between u and some analytic/"ideal"/"expected" u_reference """
        if self.u_reference.shape[0] != self.base.n_roi[0]:
            self.u_computed = self.u_computed[tuple([slice(0, self.base.n_roi[i]) for i in range(self.base.n_dims)])]
        self.rel_err = relative_error(self.u_computed, self.u_reference)
        print('Relative error: {:.2e}'.format(self.rel_err))
        return self.rel_err

    def log_and_plot(self):
        """ Call logging and plotting functions """
        self.log_details()
        self.plot_details()

    def log_details(self):
        """ Save some parameters and stats """
        print('Saving stats...')
        save_string = f'n_dims {self.base.n_dims}; boundaries width {self.base.boundary_widths}; n_domains {self.base.n_domains}; overlap {self.base.overlap}'
        if self.base.wrap_correction:
            save_string += f'; wrap correction {self.base.wrap_correction}; corner points {self.base.cp}'
        save_string += f'; {self.state.sim_time:>2.2f} sec; {self.state.iterations + 1} iterations; final residual {self.state.full_residuals[self.state.iterations]:>2.2e}'
        if hasattr(self, 'rel_err'):
            save_string += f'; relative error {self.rel_err:>2.2e}'
        save_string += f' \n'
        with open(self.stats_file_name, 'a') as fileopen:
            fileopen.write(save_string)

    def plot_details(self):
        """ Select and execute plotting functions """
        print('Plotting...')

        if self.base.n_dims == 1:
            self.x = np.arange(self.base.n_roi[0]) * self.base.pixel_size
            self.plot_field_n_residual()  # png
            self.plot_field_iters()		# movie/animation/GIF
        elif self.base.n_dims == 2:
            self.image_field_n_residual()  # png
        elif self.base.n_dims == 3:
            for z_slice in [0, int(self.u_computed.shape[2] / 2), int(self.u_computed.shape[2] - 1)]:
                self.image_field_n_residual(z_slice)  # png
        plt.close('all')
        print('Plotting done.')

    def plot_basics(self, plt_common):
        """ Plot things common to all """
        if self.base.total_domains > 1:
            plt_common.axvline(
                x=(self.base.domain_size[0]-self.base.boundary_widths[0]-self.base.overlap[0])*self.base.pixel_size,
                c='b', ls='dashdot', lw=1.5)
            plt_common.axvline(
                x=(self.base.domain_size[0]-self.base.boundary_widths[0])*self.base.pixel_size,
                c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
            for i in range(1, self.base.total_domains - 1):
                plt_common.axvline(
                    x=((i + 1) * (self.base.domain_size[0] - self.base.overlap[0]) - self.base.boundary_widths[
                        0]) * self.base.pixel_size, c='b', ls='dashdot', lw=1.5)
                plt_common.axvline(x=(i * (self.base.domain_size[0] - self.base.overlap[0]) + self.base.domain_size[0] -
                                      self.base.boundary_widths[0]) * self.base.pixel_size, c='b', ls='dashdot', lw=1.5)
        if hasattr(self, 'u_reference'):
            plt_common.plot(self.x, np.real(self.u_reference), 'k', lw=2., label=self.label)
        plt_common.ylabel('Amplitude')
        plt_common.xlabel("$x~[\lambda]$")
        plt_common.xlim([self.x[0] - self.x[1] * 2, self.x[-1] + self.x[1] * 2])
        plt_common.grid()

    def plot_field_n_residual(self):
        """ Plot the (1D) final field and residual wrt iterations and save as png """
        plt.subplots(figsize=figsize, ncols=1, nrows=2)

        plt.subplot(2, 1, 1)
        self.plot_basics(plt)
        plt.plot(self.x, np.real(self.u_computed), 'r', lw=1., label='AnySim')
        title = 'Field'
        if hasattr(self, 'u_reference'):
            plt.plot(self.x, np.real(self.u_reference - self.u_computed) * 10, 'g', lw=1., label='Error*10')
            title += f' (Relative Error = {self.rel_err:.2e})'
        plt.title(title)
        plt.legend(ncols=2, framealpha=0.6)

        plt.subplot(2, 1, 2)
        res_plots = plt.loglog(np.arange(1, self.state.iterations + 2, self.base.iter_step),
                               self.state.subdomain_residuals, lw=1.5)
        if self.base.total_domains > 1:
            plt.legend(handles=iter(res_plots), labels=tuple(f'{i + 1}' for i in range(self.base.total_domains)),
                       title='Subdomains', ncols=int(self.base.n_domains[0] / 4) + 1, framealpha=0.5)
        plt.loglog(np.arange(1, self.state.iterations+2, self.base.iter_step), self.state.full_residuals, lw=3., c='k',
                   ls='dashed', label='Full Residual')
        plt.axhline(y=self.base.threshold_residual, c='k', ls=':')
        plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        plt.ylim([y_min, y_max])
        plt.title('Residual. Iterations = {:.2e}'.format(self.state.iterations + 1))
        plt.ylabel('Residual')
        plt.xlabel('Iterations')
        plt.grid()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries ({self.base.boundary_widths}). '
        if self.base.wrap_correction:
            title_text += f'{title_text} Wrap correction: {self.base.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations + 1}iters_FieldNResidual'
        if self.base.wrap_correction == 'L_corr':
            fig_name += f'_cp{self.base.cp}'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')

    def plot_field_iters(self):  # movie/animation/GIF
        """ Plot an animation of the field wrt iterations and save as mp4 """
        self.state.u_iter = np.real(self.state.u_iter)

        fig = plt.figure(figsize=(14.32, 8))
        self.plot_basics(plt)
        line, = plt.plot([], [], 'r', lw=2., animated=True, label='AnySim')
        line.set_xdata(self.x)
        title = plt.title('')
        plt.legend()

        # Plot 100 or fewer frames. Takes much longer for any more frames.
        if self.state.iterations > 100:
            plot_iters = 100
            iters_trunc = np.linspace(0, self.state.iterations - 1, plot_iters).astype(int)
            u_iter_trunc = self.state.u_iter[iters_trunc]
        else:
            plot_iters = self.state.iterations
            iters_trunc = np.arange(self.state.iterations)
            u_iter_trunc = self.state.u_iter

        def animate(i):
            line.set_ydata(u_iter_trunc[i])  # update the data.
            title_text = f'Iteration {iters_trunc[i] + 1}. '
            title_text = f'{title_text} Absorbing boundaries ({self.base.boundary_widths}). '
            if self.base.wrap_correction:
                title_text += f'{title_text} Wrap correction: {self.base.wrap_correction}. '
            title.set_text(title_text)
            return line, title,

        ani = animation.FuncAnimation(
            fig, animate, interval=100, blit=True, frames=plot_iters)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'))
        ani_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations + 1}iters_Field'
        if self.base.wrap_correction == 'L_corr':
            ani_name += f'_cp{self.base.cp}'
        ani_name += f'.mp4'
        ani.save(ani_name, writer=writer)
        plt.close('all')

    def image_field_n_residual(self, z_slice=0):  # png
        """ Plot the (2D) final field (or 3D slice) and residual wrt iterations and save as png """
        if self.base.n_dims == 3:
            u = self.u_computed[:, :, z_slice]
            if hasattr(self, 'u_reference'):
                u_reference = self.u_reference[:, :, z_slice]
        else:
            u = self.u_computed.copy()
            if hasattr(self, 'u_reference'):
                u_reference = self.u_reference.copy()

        if hasattr(self, 'u_reference'):
            n_rows = 2
            v_lim = np.maximum(np.max(np.abs(np.real(u_reference))), np.max(np.abs(np.real(u_reference))))
        else:
            n_rows = 1
            v_lim = np.maximum(np.max(np.abs(np.real(u))), np.max(np.abs(np.real(u))))
        plt.subplots(figsize=figsize, ncols=2, nrows=n_rows)
        pad = 0.03
        shrink = 0.65

        plt.subplot(2, 2, 1)
        plt.imshow(np.real(u), cmap='seismic', vmin=-v_lim, vmax=v_lim)
        plt.colorbar(shrink=shrink, pad=pad)
        plt.title('AnySim')

        plt.subplot(2, 2, 2)
        plt.loglog(np.arange(1, self.state.iterations+2, self.base.iter_step), self.state.full_residuals, lw=3., c='k',
                   ls='dashed')
        plt.axhline(y=self.base.threshold_residual, c='k', ls=':')
        plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        plt.ylim([y_min, y_max])
        plt.title('Residual. Iterations = {:.2e}'.format(self.state.iterations + 1))
        plt.ylabel('Residual')
        plt.xlabel('Iterations')
        plt.grid()

        if hasattr(self, 'u_reference'):
            plt.subplot(2, 2, 3)
            plt.imshow(np.real(u_reference), cmap='seismic', vmin=-v_lim, vmax=v_lim)
            plt.colorbar(shrink=shrink, pad=pad)
            plt.title(self.label)

            plt.subplot(2, 2, 4)
            plt.imshow(np.real(u_reference - u), cmap='seismic')
            plt.colorbar(shrink=shrink, pad=pad)
            plt.title(f'Difference. Relative error {self.rel_err:.2e}')

        plt.tight_layout()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries ({self.base.boundary_widths}). '
        if self.base.wrap_correction:
            title_text += f'{title_text} Wrap correction: {self.base.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations + 1}iters_FieldNResidual_{z_slice}'
        if self.base.wrap_correction == 'L_corr':
            fig_name += f'_cp{self.base.cp}'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')
