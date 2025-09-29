import numpy as np
import matplotlib.pyplot as plt


N = 129
Δ = 0.25
k_coarse = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, Δ))
k_fine = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(100 * N + 1, Δ / 5))
w = 1 / (k_coarse[1] - k_coarse[0])
k0 = k_coarse[len(k_coarse) // 5]
# select_domain = np.sinc(k * Δ / np.pi)
# pm_convolve = np.convolve(select_domain, select_domain, mode="same")
# kernel = 1 / (k**2 - k0**2 + 1.0j)
M_correct = np.zeros((N, N), dtype=np.complex128)
kernel = 1.0 / (k_fine**2 - k0**2 + 0.1j * k0**2)
kernel_coarse = 1.0 / (k_coarse**2 - k0**2 + 0.1j * k0**2)

for i, p in enumerate(k_coarse):
    for j, k in enumerate(k_coarse):
        integrand = np.sinc((p - k_fine) * w) * np.sinc((k - k_fine) * w) * kernel
        M_correct[i, j] = np.trapezoid(integrand)

# analytic solution (Kirchoff)
# shift kernels by N/2-1/2
shift_exp = np.exp(1.0j * k_coarse * Δ * (N // 2 + 0.5))
edge_value_kernel = kernel_coarse * shift_exp
edge_diff_kernel = 1.0j * k_coarse * edge_value_kernel
M_diff = M_correct.copy()
np.fill_diagonal(M_diff, 0.0)
M_wrap = edge_value_kernel[:, None] * edge_diff_kernel[None, :] + edge_diff_kernel[:, None] * edge_value_kernel[None, :]
plt.subplot(1, 2, 1)
plt.imshow(M_diff.imag)
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(M_wrap.imag)
plt.show(block=True)

# find the best approximation to approximate M_correct with a diagonal and a low-rank matrix
d_approx = np.diag(M_correct)
d_correct = d_approx.copy()

M_norm = np.linalg.norm(M_correct, "fro")
order = 2
for iteration in range(150):
    # remove the diagonal part of the approximation
    M_approx = M_correct.copy()
    np.fill_diagonal(M_approx, d_correct - d_approx)

    # compute low-rank approximation of the remainder, and add back diagonal part
    U, S, V = np.linalg.svd(M_approx, full_matrices=False)
    print(f"relative error {sum((S**2)[order:]) / M_norm}")
    M_approx = U[:, :order] @ np.diag(S[:order]) @ V[:order, :] + np.diag(d_approx)

    # adjust d_approx to take into account the diagonal of the error of the low-rank approximation
    d_approx = d_approx + 0.1 * (d_correct - np.diag(M_approx))


# plt.plot(U[:, 0].real)
# plt.plot(U[:, 0].imag)
# plt.plot(edge_value_kernel.real)
# plt.plot(edge_value_kernel.imag)
# plt.legend(["U_real", "U_imag", "edge_real", "edge_imag"])
# plt.show(block=True)

plt.plot(np.angle(U[:, 0] / edge_value_kernel))
plt.show(block=True)

plt.imshow((M_correct - np.diag(np.diag(M_correct))).real)
plt.colorbar()
plt.show(block=True)
#
# plt.plot(S)
# plt.show()
print(
    np.vdot(edge_value_kernel, edge_diff_kernel) / np.linalg.norm(edge_value_kernel) / np.linalg.norm(edge_diff_kernel)
)

plt.plot(np.fft.ifftshift(np.fft.ifft(U[:, 0])))
plt.plot(-1.0j * np.fft.ifftshift(np.fft.ifft(V[0, :])))
plt.show(block=True)


a = 1.0 / d_approx * (np.pow(-1, np.round(k_coarse * w)))
plt.plot(np.angle((U[:, 0] / V[0, :])) / np.pi)
plt.plot(np.angle((U[:, 1] / V[1, :])) / np.pi)
plt.legend()
plt.show(block=True)
# plt.plot((a * a * U[:, 0] * V[0, :]).real)
# plt.plot((a * a * U[:, 0] * V[0, :]).imag)
plt.plot(d_approx.real)
plt.plot(d_correct.real)
plt.plot(d_approx.imag)
plt.plot(d_correct.imag)

plt.plot(np.angle(d_approx))
plt.plot(np.angle(d_correct))
plt.show(block=True)

plt.plot((U[:, 0] / V[0, :]).imag)
plt.show(block=True)


plt.plot((a * U[:, 0]).real)
plt.plot((a * U[:, 0]).imag)
plt.plot((a * U[:, 1]).real)
plt.plot((a * U[:, 1]).imag)
plt.show(block=True)


fU0 = np.fft.ifft(U[:, 0])
fU1 = np.fft.ifft(U[:, 1])
plt.plot(fU0.real)
plt.plot(fU0.imag)
plt.plot(fU1.real)
plt.plot(fU1.imag)
plt.show(block=True)

plt.plot(np.abs(fU0 + fU1))
plt.plot(np.abs(fU0 - fU1))
plt.show(block=True)

f_value = np.fft.ifft(edge_value_kernel)
f_diff = np.fft.ifft(edge_diff_kernel)
ratio = np.vdot(f_value[: N // 2], f_diff[: N // 2]) / np.linalg.norm(f_value[: N // 2]) ** 2
f_right = f_value * ratio - f_diff
plt.plot((f_value * ratio).real)
plt.plot((f_value * ratio).imag)
plt.plot(f_diff.real)
plt.plot(f_diff.imag)
plt.plot(np.abs(f_right))
plt.show(block=True)

a = 1.0 / d_correct
