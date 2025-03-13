Conclusion
==========

In this work :cite:`mache2024domain`, we have introduced a domain decomposition of the modified Born series (MBS) approach :cite:`osnabrugge2016convergent, vettenburg2023universal` applied to the Helmholtz equation. With the new framework, we simulated a complex 3D structure of a remarkable :math:`3.1\cdot 10^7` wavelengths in size in just :math:`1.4` hours by solving over two GPUs. This represents a factor of :math:`1.93` increase over the largest possible simulation on a single GPU without domain decomposition. 

Our decomposition framework hinges on the ability to split the linear system as :math:`A=L+V`. Instead of the traditional splitting, where :math:`V` is a scattering potential that acts locally on each voxel, we introduced a :math:`V` that includes the communication between subdomains and corrections for wraparound artefacts. As a result, the operator :math:`(L+I)^{-1}` in the MBS iteration can be evaluated locally on each subdomain using a fast convolution. Therefore, this operator, which is the most computationally intensive step of the iteration, can be evaluated in parallel on multiple GPUs. 

Despite the significant overhead of our domain splitting method due to an increased number of iterations, and communication and synchronisation overhead, the ability to split a simulation over multiple GPUs results in a very significant speedup. Already, with the current dual-GPU system, we were able to solve a problem of :math:`315\times 315\times 315` wavelengths :math:`13.2\times` faster than without domain decomposition since the non-decomposed problem is too large to fit on a single GPU. Moreover, there is only a little overhead associated with adding more subdomains along an axis after the first splitting. This favourable scaling paves the way for distributing simulations over more GPUs or compute nodes in a cluster.

In this work, we have already introduced strategies to reduce the overhead of the domain decomposition through truncating corrections to only a few points close to the edge of the subdomain and only activating certain subdomains in the iteration. We anticipate that further developments and optimisation of the code may help reduce the overhead of the lock-step execution. 

Finally, due to the generality of our approach, we expect it to be readily extended to include Maxwell's equations :cite:`kruger2017solution` and birefringent media :cite:`vettenburg2019calculating`. Given the rapid developments of GPU hardware and compute clusters, we anticipate that optical simulations at a cubic-millimetre scale can soon be performed in a matter of minutes.

Code availability
-----------------
The code for Wavesim is available on GitHub :cite:`wavesim_py`, it is licensed under the MIT license. When using Wavesim in your work, please cite :cite:`mache2024domain, osnabrugge2016convergent` and this current paper. Examples and documentation for this project are available at `Read the Docs <https://wavesim.readthedocs.io/en/latest/>`_ :cite:`wavesim_documentation`.

%endmatter%