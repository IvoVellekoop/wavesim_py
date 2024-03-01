from helmholtzbase import HelmholtzBase
from state import State
from utilities import max_abs_error, relative_error

import os
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc

font = {'family': 'Times New Roman',  # 'Times New Roman', 'Helvetica', 'Arial', 'Cambria', or 'Symbol'
        'size': 10}  # 8-10 pt
rc('font', **font)
figsize = (8, 8)  # (14.32,8)
plt.rcParams['text.usetex'] = True


class LogPlot:
    def __init__(self, base: HelmholtzBase, state: State, u_computed: np.array([]), u_reference=None,
                 animate_iters=False):
        """ Logging and Plotting Class """
        self.base = base
        self.state = state
        self.u_computed = u_computed.cpu().numpy()
        if u_reference is not None:
            self.u_reference = u_reference
        self.animate_iters = animate_iters
        self.truncate_iterations = False
        self.rel_err = None
        self.mae = None
        if hasattr(self, 'u_reference'):
            self.compare()  # relative error between u and u_true
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
            self.run_id += '_' + self.base.wrap_correction + f'{self.base.n_correction}'
        self.run_id += '_n_domains' + str(self.base.n_domains)

        self.stats_file_name = self.run_loc + '_stats.txt'

        if hasattr(self, 'u_reference'):
            self.label = 'Reference solution'

        if self.animate_iters:
            # self.u_iter = self.state.u_iter.copy()
            self.u_iter = self.state.u_iter.cpu().numpy()
            self.plot_iter_step = 1
            if self.state.iterations*self.base.total_domains > 100:
                self.plot_iter_step = int(np.ceil((self.state.iterations*self.base.total_domains)/100))
                self.truncate_iterations = True
                for i in self.base.domains_iterator:
                    self.u_iter[i] = self.u_iter[i][::self.plot_iter_step]
            self.u_iter = np.array(list(map(list, self.u_iter.values())))   # convert dict of lists to array

    def compare(self):
        """ Compute relative error between computed and reference field """
        if self.u_reference.shape[0] != self.base.n_roi[0]:
            self.u_computed = self.u_computed[tuple([slice(0, self.base.n_roi[i]) for i in range(self.base.n_dims)])]
        self.rel_err = relative_error(self.u_computed, self.u_reference)
        self.mae = max_abs_error(self.u_computed, self.u_reference)

    def log_and_plot(self):
        """ Call logging and plotting functions """
        self.log_details()
        self.plot_details()

    def log_details(self):
        """ Save parameters and stats """
        print('Saving stats...')
        save_string = (f'n_dims {self.base.n_dims}; boundaries width {self.base.boundary_widths}; '
                       + f'n_domains {self.base.n_domains}')
        if self.base.wrap_correction:
            save_string += f'; {self.base.wrap_correction}; n_correction {self.base.n_correction}'
        if self.base.total_domains > 1:
            save_string += f'; n_transfer {self.base.n_correction}'
        save_string += (f'; {self.state.sim_time:>2.2f} sec; {self.state.iterations} iterations; '
                        + f'final residual {self.state.full_residuals[self.state.iterations-1]:>2.2e}')
        if self.rel_err:
            save_string += f'; relative error {self.rel_err:>2.2e}'
        if self.mae:
            save_string += f'; max abs error (normalized) {self.mae:>2.2e}'
        save_string += f' \n'
        with open(self.stats_file_name, 'a') as fileopen:
            fileopen.write(save_string)

    def plot_details(self):
        """ Select and execute plotting functions """
        print('Plotting...')

        if self.base.n_dims == 1:
            self.x = np.arange(self.base.n_roi[0]) * self.base.pixel_size
            self.plot_field_n_residual()  # png
            if self.animate_iters:
                self.anim_field_n_residual_1d()  # movie/animation/GIF
        elif self.base.n_dims == 2:
            self.image_field_n_residual()  # png
            if self.animate_iters:
                self.anim_field_n_residual()  # movie/animation/GIF
        elif self.base.n_dims == 3:
            for idx, z_slice in enumerate([0, int(self.u_computed.shape[2] / 2), int(self.u_computed.shape[2] - 1)]):
                self.image_field_n_residual(z_slice)  # png
                if self.animate_iters:
                    self.anim_field_n_residual(idx, z_slice)  # movie/animation/GIF
        plt.close('all')
        print('Plotting done.')

    def plot_common_things(self, plt_common):
        """ Plot things common to all """
        if self.base.total_domains > 1:
            plt_common.axvline(x=(self.base.domain_size[0] - self.base.boundary_widths[0]) * self.base.pixel_size, 
                               c='b', ls='dashdot', lw=1.5)
            plt_common.axvline(x=(self.base.domain_size[0] - self.base.boundary_widths[0]) * self.base.pixel_size, 
                               c='b', ls='dashdot', lw=1.5, label='Subdomain boundaries')
            for i in range(1, self.base.total_domains - 1):
                plt_common.axvline(x=(i * self.base.domain_size[0] + self.base.domain_size[0] 
                                      - self.base.boundary_widths[0]) * self.base.pixel_size, 
                                   c='b', ls='dashdot', lw=1.5)
        if hasattr(self, 'u_reference'):
            plt_common.plot(self.x, np.real(self.u_reference), 'k', lw=2., label=self.label)
        plt_common.set_ylabel('Amplitude')
        plt_common.set_xlabel("$x~[\lambda]$")
        plt_common.set_xlim([self.x[0] - self.x[1] * 2, self.x[-1] + self.x[1] * 2])
        plt_common.grid()

    def plot_field_n_residual(self):
        """ Plot the (1D) final field and residual wrt iterations and save as png """
        fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=2)
        ax = ax.flatten()

        self.plot_common_things(ax[0])
        ax[0].plot(self.x, np.real(self.u_computed), 'r', lw=1., label='AnySim')
        title = 'Field'
        if hasattr(self, 'u_reference'):
            ax[0].plot(self.x, np.abs(self.u_reference - self.u_computed)*10, 'g', lw=1., label='Error*10')
            title += f' (Rel Err = {self.rel_err:.2e}, MAE = {self.mae:.2e})'
        # ax[0].axvspan(0*self.base.pixel_size, 99*self.base.pixel_size, facecolor='lightgrey', alpha=0.5, label='n=1')
        ax[0].set_title(title)
        ax[0].legend(ncols=2, framealpha=0.8)

        res_plots = ax[1].loglog(np.arange(1, self.state.iterations+1),
                                 self.state.subdomain_residuals.cpu().numpy(), lw=1.5)
        if self.base.total_domains > 1:
            ax[1].legend(handles=iter(res_plots), labels=tuple(f'{i + 1}' for i in range(self.base.total_domains)),
                         title='Subdomains', ncols=int(self.base.total_domains/4)+1, framealpha=0.5)
        ax[1].loglog(np.arange(1, self.state.iterations+1), self.state.full_residuals, lw=3., c='k',
                     ls='dashed', label='Full Residual')
        ax[1].axhline(y=self.base.threshold_residual, c='k', ls=':')
        ax[1].set_yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        ax[1].set_ylim([y_min, y_max])
        ax[1].set_title(f'Residual {self.state.full_residuals[-1]:.2e}. Iterations = {self.state.iterations:.2e}')
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('Iterations')
        ax[1].grid()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries {self.base.boundary_widths}. '
        if self.base.wrap_correction:
            title_text = f'{self.base.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations}iters_FieldNResidual'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')

    def anim_field_iters(self):  # movie/animation/GIF
        """ Plot an animation of the field wrt iterations and save as mp4 """
        u_iter = np.real(self.u_iter)
        u_iter = np.reshape(u_iter, [-1, u_iter.shape[2]], 'F')

        fig = plt.figure(figsize=(14.32, 8))
        self.plot_common_things(plt)
        plot_data, = plt.plot([], [], 'r', lw=2., animated=True, label='AnySim')
        plot_data.set_xdata(self.x)
        plt.legend(ncols=int(self.base.total_domains/4)+1, framealpha=0.5)
        title = plt.title('')

        # Plot 100 or fewer frames. Takes much longer for any more frames.
        if self.state.iterations*self.base.total_domains > 100:
            plot_iters = int(self.state.iterations*self.base.total_domains/10)
            iters_trunc = np.linspace(0, self.state.iterations*self.base.total_domains - 1, plot_iters).astype(int)
            domains_trunc = self.base.domains_iterator * plot_iters
            u_iter_trunc = u_iter[iters_trunc]
        else:
            plot_iters = self.state.iterations * self.base.total_domains
            iters_trunc = np.arange(self.state.iterations)
            domains_trunc = self.base.domains_iterator * self.state.iterations
            u_iter_trunc = u_iter

        def animate(i):
            plot_data.set_ydata(u_iter_trunc[i])  # update the data.
            title_text = f'Iteration {iters_trunc[i//self.base.total_domains] + 1}, Subdomain {domains_trunc[i]}. '
            title.set_text(title_text)
            return plot_data, title,

        ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, frames=plot_iters)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'))
        ani_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations}iters_Field'
        ani_name += f'.mp4'
        ani.save(ani_name, writer=writer)
        plt.close('all')

    def anim_field_n_residual_1d(self):  # movie/animation/GIF
        """ Plot an animation of the field and residual wrt iterations and save as mp4 """
        u_iter = np.real(self.u_iter)
        u_iter = np.reshape(u_iter, [-1, u_iter.shape[2]], 'F')

        fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=2)
        ax = ax.flatten()

        plt.subplot(2, 1, 1)
        self.plot_common_things(ax[0])
        ax[0].plot([], [], 'r', lw=2., animated=True, label='AnySim')
        ax[0].legend(ncols=int(self.base.total_domains/4)+1, framealpha=0.5)

        if self.truncate_iterations:
            plot_iters = len(u_iter)
            iters_trunc = np.arange(0, self.state.iterations, self.plot_iter_step)
            residuals = self.state.full_residuals.cpu().numpy()[::self.plot_iter_step]
            subdomain_residuals = self.state.subdomain_residuals.cpu().numpy()[::self.plot_iter_step, :]
        else:
            plot_iters = self.state.iterations * self.base.total_domains
            iters_trunc = np.arange(self.state.iterations)
            # residuals = self.state.full_residuals.copy()
            residuals = self.state.full_residuals.cpu().numpy()
            # subdomain_residuals = self.state.subdomain_residuals.copy()
            subdomain_residuals = self.state.subdomain_residuals.cpu().numpy()
        domains_trunc = self.base.domains_iterator * self.state.iterations

        frames = []
        for i in range(plot_iters):
            line0, = ax[0].plot(self.x, u_iter[i], 'r', lw=2., animated=True)
            text0 = ax[0].text(0.35, 1.01,
                               f'Iteration {iters_trunc[i//self.base.total_domains]+1}, Subdomain {domains_trunc[i]}.',
                               ha="left", va="bottom", transform=ax[0].transAxes)

            line1, = ax[1].loglog(iters_trunc[:i//self.base.total_domains+1]+1,
                                  residuals[:i//self.base.total_domains+1], lw=2., c='k', label='Full Residual')
            if self.base.total_domains > 1:
                lines2 = ax[1].loglog(iters_trunc[:i//self.base.total_domains+1]+1,
                                      subdomain_residuals[:i//self.base.total_domains+1, :], lw=1.5)
            else:
                lines2 = []

            frames.append([line0, text0, line1] + lines2)

        if self.base.total_domains > 1:
            ax[1].legend(handles=iter(lines2), labels=tuple(f'{i + 1}' for i in range(self.base.total_domains)),
                         title='Subdomains', ncols=int(self.base.total_domains/4)+1, framealpha=0.5)
        ax[1].axhline(y=self.base.threshold_residual, c='k', ls=':')
        ax[1].set_yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        ax[1].set_title('Residual')
        ax[1].set_ylim([y_min, y_max])
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('Iterations')
        ax[1].grid()
        plt.tight_layout()

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'))
        ani_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations}iters_Field'
        ani_name += f'.mp4'
        ani.save(ani_name, writer=writer)
        plt.close('all')

    def image_field_n_residual(self, z_slice=0):  # png
        """ Plot the (2D) final field (or 3D slice) and residual wrt iterations and save as png """
        if self.base.n_dims == 3:
            u = np.abs(self.u_computed[:, :, z_slice])
            if hasattr(self, 'u_reference'):
                u_reference = np.abs(self.u_reference[:, :, z_slice])
        else:
            u = np.abs(self.u_computed)
            if hasattr(self, 'u_reference'):
                u_reference = np.abs(self.u_reference)

        if hasattr(self, 'u_reference'):
            n_rows = 2
            v_lim = np.maximum(np.max(u_reference), np.max(u_reference))
        else:
            n_rows = 1
            v_lim = np.maximum(np.max(u), np.max(u))
        plt.subplots(figsize=figsize, ncols=2, nrows=n_rows)
        pad = 0.03
        shrink = 0.65

        plt.subplot(2, 2, 1)
        plt.imshow(u, cmap='seismic', vmin=-v_lim, vmax=v_lim)
        plt.colorbar(shrink=shrink, pad=pad)
        plt.title('AnySim')

        plt.subplot(2, 2, 2)
        res_plots = plt.loglog(np.arange(1, self.state.iterations+1),
                               self.state.subdomain_residuals.cpu().numpy(), lw=1.5)
        if self.base.total_domains > 1:
            plt.legend(handles=iter(res_plots), labels=tuple(f'{i + 1}' for i in range(self.base.total_domains)),
                       title='Subdomains', ncols=int(self.base.total_domains/4)+1, framealpha=0.5)
        plt.loglog(np.arange(1, self.state.iterations+1), self.state.full_residuals, lw=3., c='k',
                   ls='dashed')
        plt.axhline(y=self.base.threshold_residual, c='k', ls=':')
        plt.yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        plt.ylim([y_min, y_max])
        plt.title(f'Residual {self.state.full_residuals[-1]:.2e}. Iterations {self.state.iterations:.2e}')
        plt.ylabel('Residual')
        plt.xlabel('Iterations')
        plt.grid()

        if hasattr(self, 'u_reference'):
            plt.subplot(2, 2, 3)
            im3 = plt.imshow(u_reference, cmap='seismic', vmin=-v_lim, vmax=v_lim)
            plt.colorbar(mappable=im3, shrink=shrink, pad=pad)
            plt.title(self.label)

            plt.subplot(2, 2, 4)
            im4 = plt.imshow(u_reference - u, cmap='seismic')
            plt.colorbar(mappable=im4, shrink=shrink, pad=pad)
            plt.title(f'Difference. Rel Err {self.rel_err:.2e}. MAE {self.mae:.2e}')

        plt.tight_layout()

        title_text = ''
        title_text = f'{title_text} Absorbing boundaries {self.base.boundary_widths}. '
        if self.base.wrap_correction:
            title_text = f'{title_text} {self.base.wrap_correction}. '
        plt.suptitle(title_text)

        plt.tight_layout()
        fig_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations}iters_FieldNResidual_{z_slice}'
        fig_name += f'.png'
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=100)
        plt.close('all')

    def anim_field_n_residual(self, idx=0, z_slice=0):  # movie/animation/GIF
        """ Plot an animation of the (2D/3D) field image and residual wrt iterations and save as mp4 """
        if self.base.n_dims == 3:
            u_iter = self.u_iter[..., idx]
        else:
            u_iter = self.u_iter.copy()
        u_iter = np.reshape(u_iter, [-1, u_iter.shape[2], u_iter.shape[3]], 'F')

        if hasattr(self, 'u_reference'):
            if self.base.n_dims == 3:
                u_reference = np.abs(self.u_reference[:, :, z_slice])
            else:
                u_reference = np.abs(self.u_reference)
            n_rows = 2
            v_lim = np.maximum(np.max(u_reference), np.max(u_reference))
        else:
            n_rows = 1
            v_lim = np.maximum(np.max(u_iter), np.max(u_iter))

        fig, ax = plt.subplots(figsize=figsize, ncols=2, nrows=n_rows)
        ax = ax.flatten()
        pad = 0.03
        shrink = 0.7

        if self.truncate_iterations:
            plot_iters = len(u_iter)
            iters_trunc = np.arange(0, self.state.iterations, self.plot_iter_step)
            residuals = self.state.full_residuals.cpu().numpy()[::self.plot_iter_step]
            subdomain_residuals = self.state.subdomain_residuals.cpu().numpy()[::self.plot_iter_step, :]
        else:
            plot_iters = self.state.iterations * self.base.total_domains
            iters_trunc = np.arange(self.state.iterations)
            # residuals = self.state.full_residuals.copy()
            residuals = self.state.full_residuals.cpu().numpy()
            # subdomain_residuals = self.state.subdomain_residuals.copy()
            subdomain_residuals = self.state.subdomain_residuals.cpu().numpy()
        domains_trunc = self.base.domains_iterator * self.state.iterations

        frames = []
        for i in range(plot_iters):
            im0 = ax[0].imshow(u_iter[i], cmap='seismic', vmin=-v_lim, vmax=v_lim, animated=True)
            text0 = ax[0].text(0.1, 1.01,
                               f'Iteration {iters_trunc[i//self.base.total_domains]+1}, Subdomain {domains_trunc[i]}.',
                               ha="left", va="bottom", transform=ax[0].transAxes)

            line1, = ax[1].loglog(iters_trunc[:i//self.base.total_domains+1]+1,
                                  residuals[:i//self.base.total_domains+1], lw=2., c='k', label='Full Residual')
            if self.base.total_domains > 1:
                lines2 = ax[1].loglog(iters_trunc[:i//self.base.total_domains+1]+1,
                                      subdomain_residuals[:i//self.base.total_domains+1, :], lw=1.5)
            else:
                lines2 = []

            if hasattr(self, 'u_reference'):
                im3 = ax[3].imshow(u_reference - u_iter[i], cmap='seismic', vmin=-v_lim, vmax=v_lim, animated=True)
                frames.append([im0, text0, line1, im3] + lines2)
            else:
                frames.append([im0, text0, line1] + lines2)

        plt.colorbar(mappable=im0, ax=ax[0], shrink=shrink, pad=pad)
        if self.base.total_domains > 1:
            ax[1].legend(handles=iter(lines2), labels=tuple(f'{i + 1}' for i in range(self.base.total_domains)),
                         title='Subdomains', ncols=int(self.base.total_domains/4)+1, framealpha=0.5)
        ax[1].axhline(y=self.base.threshold_residual, c='k', ls=':')
        ax[1].set_yticks([1.e+6, 1.e+3, 1.e+0, 1.e-3, 1.e-6, 1.e-9, 1.e-12])
        y_min = np.minimum(6.e-7, 0.8 * np.nanmin(self.state.subdomain_residuals))
        y_max = np.maximum(2.e+0, 1.2 * np.nanmax(self.state.subdomain_residuals))
        ax[1].set_title('Residual')
        ax[1].set_ylim([y_min, y_max])
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('Iterations')
        ax[1].grid()
        if hasattr(self, 'u_reference'):
            im2 = ax[2].imshow(u_reference, cmap='seismic', vmin=-v_lim, vmax=v_lim)
            plt.colorbar(mappable=im2, ax=ax[2], shrink=shrink, pad=pad)
            ax[2].set_title(self.label)
            plt.colorbar(mappable=im3, ax=ax[3], shrink=shrink, pad=pad)
            ax[3].set_title(f'Difference. Rel Err {self.rel_err:.2e}. MAE {self.mae:.2e}')
        plt.tight_layout()

        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
        writer = animation.FFMpegWriter(fps=10, metadata=dict(artist='Me'))
        ani_name = f'{self.run_loc}/{self.run_id}_{self.state.iterations}iters_Field'
        if self.base.n_dims == 3:
            ani_name += f'_zslice{z_slice}'
        ani_name += f'.mp4'
        ani.save(ani_name, writer=writer)
        plt.close('all')
