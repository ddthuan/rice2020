# https://laurentperrinet.github.io/sciblog/posts/2017-09-20-the-fastest-2d-convolution-in-the-world.html

from __future__ import division, print_function

from libcore import *

#np.set_printoptions(precision=6, suppress=True)
import os

phi = (np.sqrt(5)+1)/2
fig_width = 10
figsize = (fig_width, fig_width/phi)

N, n_N, M, n_M = 1024, 3, 3, 3

def get_data(N=N, n_N=n_N, M=M, n_M=n_M, seed=42, prefetching=False):
    np.random.seed(seed)
    if prefetching:
        A = np.random.rand(n_N, N, N)
        B = np.random.rand(n_M, M, M)
        C = np.zeros((n_N, n_M, N, N))
    else:
        A = np.random.rand(N, N, n_N)
        B = np.random.rand(M, M, n_M)
        C = np.zeros((N, N, n_N, n_M))
    return A, B, C

def test_get_data(N=N, n_N=n_N, M=M, n_M=n_M):
    A, B, C = get_data(N, n_N, M, n_M) 
    
def test_numpy_fft_opt(A, B, prefetching=False):
    if prefetching:
        f_B = np.zeros((B.shape[0], A.shape[-2], A.shape[-1]), dtype=np.complex128)
        for i_M in np.arange(B.shape[0]):
            f_B[i_M, :, :] = fft2(B[i_M, :, :], s=A.shape[-2:])
        
        for i_N in np.arange(A.shape[0]):
            f_A = fft2(A[i_N, :, :])
            for i_M in np.arange(B.shape[0]):
                C[i_N, i_M, :, :] = np.real(ifft2(f_A*f_B[i_M, :, :]))
    else:
        f_B = np.zeros((A.shape[0], A.shape[1], B.shape[-1]), dtype=np.complex128)
        for i_M in np.arange(B.shape[-1]):
            f_B[:, :, i_M] = fft2(B[:, :, i_M], s=A.shape[:2])
        
        for i_N in np.arange(A.shape[-1]):
            f_A = fft2(A[:, :, i_N])
            for i_M in np.arange(B.shape[-1]):
                C[:, :, i_N, i_M] = np.real(ifft2(f_A*f_B[:, :, i_M]))
                
def test_torch(A, B, prefetching=False):
# =============================================================================
#     if prefetching:
#         A = np.swapaxes(A, 0, -2)        
#         B = np.swapaxes(B, 0, -2)        
#         A = np.swapaxes(A, 1, -1)        
#         B = np.swapaxes(B, 1, -1)        
# =============================================================================
    if prefetching:
        A = Fv.to_tensor(A)
        B = Fv.to_tensor(B)
    #A = torch.from_numpy(A[:, None, :, :])
    #B = torch.from_numpy(B[:, None, :, :])
    A = A[:, None]
    B = B[:, None]
    C = F.conv2d(A, B, padding=B.shape[-1]//2)


model = torch.nn.Conv2d(1,n_M, M,padding=M//2, bias=False)
model.to("cuda:0")
#torch.backends.cudnn.deterministic = True
def test_pytorch_gpu(A, B, prefetching=False):
    if prefetching:
        A = Fv.to_tensor(A)
        B = Fv.to_tensor(B)
    A = A[:, None]
    B = B[:, None]
    
    A.to("cuda:0")
    B.to("cuda:0")
    #C = F.conv2d(A, B, padding=B.shape[-1]//2)
    model.weight.data = B
    C = model(A)
    

from timeit import Timer
reps = 0
npo, pt, pt_gpu = [], [], []


# test so luong anh
n_Ns = 2**np.arange(10)

for prefetching in [True]:
    for n_N_ in n_Ns:
        A, B, C = get_data(N, n_N_, M, n_M, prefetching=prefetching)
        
        t = Timer(lambda: test_pytorch_gpu(A, B, prefetching=prefetching))
        pt.append(t.timeit(number=reps))
                
        t = Timer(lambda: test_numpy_fft_opt(A, B, prefetching=prefetching))
        npo.append(t.timeit(number=reps))
        

fig , ax = plt.subplots(figsize=(8, 5))
ax.loglog(n_Ns, pt[:len(n_Ns)], c='c', label='torch')
ax.loglog(n_Ns, npo[:len(n_Ns)], c='m', label='numpy')

# =============================================================================
# ax.loglog(n_Ns, pt[len(n_Ns):], '--', c='c')
# ax.loglog(n_Ns, npo[len(n_Ns):], '--', c='m')
# =============================================================================

ax.set_xlabel('n_N')
ax.set_ylabel('time (s)')
ax.legend()
plt.show()


# =============================================================================
# # test so bo loc
# n_Ms = 2**np.arange(6)
# 
# for prefetching in [True]:
#     for n_M_ in n_Ms:
#         A, B, C = get_data(N, n_N, M, n_M_, prefetching=prefetching)
#         
#         t = Timer(lambda: test_pytorch_gpu(A, B, prefetching=prefetching))
#         pt.append(t.timeit(number=reps))
#                 
#         t = Timer(lambda: test_numpy_fft_opt(A, B, prefetching=prefetching))
#         npo.append(t.timeit(number=reps))
#         
# 
# fig , ax = plt.subplots(figsize=(8, 5))
# ax.loglog(n_Ms, pt[:len(n_Ms)], c='c', label='torch')
# ax.loglog(n_Ms, npo[:len(n_Ms)], c='m', label='numpy')
# 
# #ax.loglog(n_Ms, pt[len(n_Ms):], '--', c='c')
# #ax.loglog(n_Ms, npo[len(n_Ms):], '--', c='m')
# 
# ax.set_xlabel('Ms')
# ax.set_ylabel('time (s)')
# ax.legend()
# plt.show()
# =============================================================================


# =============================================================================
# # test kich thuot bo loc, test 1 image
# reps = 20
# #Ms = 2**np.arange(6)
# #Ms = [3,4,5,6,7,8,9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# Ms = [3,4,5,6,7,8,9, 10, 11, 12, 13, 14]
# for prefetching in [True]:
#     for M_ in Ms:
#         A, B, C = get_data(N, n_N, M_, n_M, prefetching=prefetching)
#         
#         t = Timer(lambda: test_numpy_fft_opt(A, B, prefetching=prefetching))
#         npo.append(t.timeit(number=reps))
#         
#         t = Timer(lambda: test_torch(A, B, prefetching=prefetching))
#         pt.append(t.timeit(number=reps))
#         
#         t = Timer(lambda: test_pytorch_gpu(A, B, prefetching=prefetching))
#         pt_gpu.append(t.timeit(number=reps))
#                 
# 
# fig , ax = plt.subplots(figsize=(8, 5))
# ax.loglog(Ms, npo[:len(Ms)], c='b', label='frequency domain, numpy')
# ax.loglog(Ms, pt[:len(Ms)], c='y', label='spatial domain, pytorch')
# ax.loglog(Ms, pt_gpu[:len(Ms)], c='r', label='spatial domain, pytorch_cuda')
# 
# ax.set_xlabel('kernel size')
# ax.set_ylabel('time (s)')
# ax.legend()
# plt.savefig('img_size1024.png')
# plt.show()
#         
# =============================================================================
                

        