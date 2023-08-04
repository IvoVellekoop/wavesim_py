from anysim_main import AnySim

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


def print_details(sim: AnySim):
    print(f'\n{sim.n_dims} dimensional problem')
    if sim.wrap_correction:
        print('Wrap correction: \t', sim.wrap_correction)
    print('Boundaries width: \t', sim.boundary_widths)
    if sim.total_domains > 1:
        print(
            f'Decomposing into {sim.n_domains} domains of size {sim.domain_size}, overlap {sim.overlap}')


# Relative error
def relative_error(e, e_true):
    return np.mean(np.abs(e - e_true) ** 2) / np.mean(np.abs(e_true) ** 2)


# Compute and print relative error between u and some analytic/"ideal"/"expected" u_true
def compare(sim: AnySim, u_true):
    if u_true.shape[0] != sim.n_roi[0]:
        sim.u = sim.u[tuple([slice(0, sim.n_roi[i]) for i in range(sim.n_dims)])]
        # sim.u_iter = sim.u_iter[:, :sim.n_roi]
    rel_err = relative_error(sim.u, u_true)
    print('Relative error: {:.2e}'.format(rel_err))
    return rel_err


class LogPlot:
    def __init__(self, sim: AnySim, u_true: np.array([]), rel_err: None):
        self.sim = sim
        self.u_true = u_true
        self.rel_err = rel_err
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

        self.run_id = d1 + '_n_dims' + str(self.sim.n_dims)
        self.run_loc = self.log_dir + self.run_id
        if not os.path.exists(self.run_loc):
            print(f'"{self.run_loc}" does not exist, creating...')
            os.makedirs(self.run_loc)
        else:
            print(f'"{self.run_loc}" exists.')

        self.run_id += '_abs' + str(self.sim.boundary_widths)
        if self.sim.wrap_correction:
            self.run_id += '_' + self.sim.wrap_correction
        self.run_id += '_n_domains' + str(self.sim.n_domains)

        self.stats_file_name = self.run_loc + '_stats.txt'

        if 'u_true' in locals():
            self.label = 'Reference solution'

    def log_and_plot(self):
        self.log_details()
        self.plot_details()
        self.plotting_done = True

    # Save some parameters and stats
    def log_details(self):
        print('Saving stats...')
        save_string = f'n_dims {self.sim.n_dims}; boundaries width {self.sim.boundary_widths}; n_domains {self.sim.n_domains}; overlap {self.sim.overlap}'
        if self.sim.wrap_correction:
            save_string += f'; wrap correction {self.sim.wrap_correction}; corner points {self.sim.cp}'
        save_string += f'; {self.sim.sim_time:>2.2f} sec; {self.sim.iterations + 1} iterations; final residual {self.sim.residual_i:>2.2e}'
        if hasattr(self, 'rel_err'):
            save_string += f'; relative error {self.rel_err:>2.2e}'
        save_string += f' \n'
        with open(self.stats_file_name, 'a') as fileopen:
            fileopen.write(save_string)

    # Plotting functions
    def plot_details(self):
        print('Plotting...')

        if self.sim.n_dims == 1:
            self.x = np.arange(self.sim.n_roi[0]) * self.sim.pixel_size
            self.plot_field_n_residual()  # png
            # self.sim.plot_field_iters()		# movie/animation/GIF
        elif self.sim.n_dims == 2:
            self.image_field_n_residual()  # png
        elif self.sim.n_dims == 3:
            for z_slice in [0, int(self.sim.u.shape[2] / 2), int(self.sim.u.shape[2] - 1)]:
                self.image_field_n_residual(z_slice)  # png
        plt.close('all')
        print('Plotting done.')

    def plot_basics(self, plt_common):
        if self.sim.total_domains > 1:
            plt_common.axvline(
                x=(self.sim.domain_size[0] - self.sim.boundary_widths[0] - self.sim.overlap[0]) * self.sim.pixel_size,
                c='b', ls='dashdot', lw=1.5)
            plt_common.axvline(x=(self.sim.domain_size[0] - self.sim.boundary_widths[0]) * self.sim.pixel_size, c='b',
                               ls='dashdot',
                               lw=1.5, label='Subdomain boundaries')
            for i in range(1, self.sim.total_domains - 1):
                plt_common.axvline(
                    x=((i + 1) * (self.sim.domain_size[0] - self.sim.overlap[0]) - self.sim.boundary_widths[
                        0]) * self.sim.pixel_size, c='b', ls='dashdot', lw=1.5)
                plt_common.axvline(x=(i * (self.sim.domain_size[0] - self.sim.overlap[0]) + self.sim.domain_size[0] -
                                      self.sim.boundary_widths[0]) * self.sim.pixel_size, c='b', ls='dashdot', lw=1.5)
        if hasattr(self, 'u_true'):
            plt_common.plot(self.x, np.real(self.u_true), 'k', lw=2., label=self.label)
        plt_common.ylabel('Amplitude')
        plt_common.xlabel("$x~[\lambda]$")
        plt_common.xlim([self.x[0] - self.x[1] * 2, self.x[-1] + self.x[1] * 2])
        plt_common.grid()

    def plot_field_n_residual(self):  # png
        plt.subplots(figsize=figsize, ncols=1, nrows=2)

        plt.subplot(2, 1, 1)
        self.plot_basics(plt)
        plt.plot(self.x, np.real(self.sim.u), 'r', lw=1., label='AnySim')
        title = 'Field'
        if hasattr(self, 'u_true'):
            plt.plot(self.x, np.real(self.u_true - self.sim.u) * 10, 'g', lw=1., label='Error*10')
            title += f' (Relative Error = {self.rel_err:.2e})'
        plt.title(title)
        plt.legend(ncols=2, framealpha=0.6)

        plt.subplot(2, 1, 2)
        res_plots = plt.loglog(np.arange(1, self.sim.iterations + 2, self.sim.iter_step), self.sim.residual, lw=1.5)
        if self.sim.total_domains > 1:
            plt.legend(handles=iter(res_plots), labels=tuple(f'{i + 1}' for i in self.sim.range_total_domains),
                       title='Subdomains', ncols=int(self.sim.n_domains[0] / 4) + 1, framealpha=0.5)
        plt.loglog(np.arange(1, self.sim.iterations + 2, self.sim.iter_step), self.sim.full_residual, lw=3., c='k',
                   ls='dashed', label='Full Residual')
        plt.axhline(y=self.sim.threshold_residual, c='k', ls=':')
        plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.sim.residual))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.sim.residual))
        plt.ylim([y_min, y_max])
        plt.title('Residual. Iterations = {:.2e}'.format(self.sim.iterations + 1))
        plt.ylabel('Residual')
        plt.xlabel('Iterations')
        plt.grid()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries ({self.sim.boundary_widths}). '
        if self.sim.wrap_correction:
            title_text += f'{title_text} Wrap correction: {self.sim.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.sim.iterations + 1}iters_FieldNResidual'
        if self.sim.wrap_correction == 'L_corr':
            fig_name += f'_cp{self.sim.cp}'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')

    def plot_field_iters(self):  # movie/animation/GIF
        self.sim.u_iter = np.real(self.sim.u_iter)

        fig = plt.figure(figsize=(14.32, 8))
        self.plot_basics(plt)
        line, = plt.plot([], [], 'r', lw=2., animated=True, label='AnySim')
        line.set_xdata(self.x)
        title = plt.title('')
        plt.legend()

        # Plot 100 or fewer frames. Takes much longer for any more frames.
        if self.sim.iterations > 100:
            plot_iters = 100
            iters_trunc = np.linspace(0, self.sim.iterations - 1, plot_iters).astype(int)
            u_iter_trunc = self.sim.u_iter[iters_trunc]
        else:
            plot_iters = self.sim.iterations
            iters_trunc = np.arange(self.sim.iterations)
            u_iter_trunc = self.sim.u_iter

        def animate(i):
            line.set_ydata(u_iter_trunc[i])  # update the data.
            title_text = f'Iteration {iters_trunc[i] + 1}. '
            title_text = f'{title_text} Absorbing boundaries ({self.sim.boundary_widths}). '
            if self.sim.wrap_correction:
                title_text += f'{title_text} Wrap correction: {self.sim.wrap_correction}. '
            title.set_text(title_text)
            return line, title,

        ani = animation.FuncAnimation(
            fig, animate, interval=100, blit=True, frames=plot_iters)
        writer = animation.FFMpegWriter(
            fps=10, metadata=dict(artist='Me'))
        ani_name = f'{self.run_loc}/{self.run_id}_{self.sim.iterations + 1}iters_Field'
        if self.sim.wrap_correction == 'L_corr':
            ani_name += f'_cp{self.sim.cp}'
        ani_name += f'.mp4'
        ani.save(ani_name, writer=writer)
        plt.close('all')

    def image_field_n_residual(self, z_slice=0):  # png
        if self.sim.n_dims == 3:
            u = self.sim.u[:, :, z_slice]
            if hasattr(self, 'u_true'):
                u_true = self.u_true[:, :, z_slice]
        else:
            u = self.sim.u.copy()
            if hasattr(self, 'u_true'):
                u_true = self.u_true.copy()

        if hasattr(self, 'u_true'):
            n_rows = 2
            v_lim = np.maximum(np.max(np.abs(np.real(u_true))), np.max(np.abs(np.real(u_true))))
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
        plt.loglog(np.arange(1, self.sim.iterations + 2, self.sim.iter_step), self.sim.full_residual, lw=3., c='k',
                   ls='dashed')
        plt.axhline(y=self.sim.threshold_residual, c='k', ls=':')
        plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.sim.residual))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.sim.residual))
        plt.ylim([y_min, y_max])
        plt.title('Residual. Iterations = {:.2e}'.format(self.sim.iterations + 1))
        plt.ylabel('Residual')
        plt.xlabel('Iterations')
        plt.grid()

        if hasattr(self, 'u_true'):
            plt.subplot(2, 2, 3)
            plt.imshow(np.real(u_true), cmap='seismic', vmin=-v_lim, vmax=v_lim)
            plt.colorbar(shrink=shrink, pad=pad)
            plt.title(self.label)

            plt.subplot(2, 2, 4)
            plt.imshow(np.real(u_true - u), cmap='seismic')
            plt.colorbar(shrink=shrink, pad=pad)
            plt.title(f'Difference. Relative error {self.rel_err:.2e}')

        plt.tight_layout()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries ({self.sim.boundary_widths}). '
        if self.sim.wrap_correction:
            title_text += f'{title_text} Wrap correction: {self.sim.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.sim.iterations + 1}iters_FieldNResidual_{z_slice}'
        if self.sim.wrap_correction == 'L_corr':
            fig_name += f'_cp{self.sim.cp}'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')
