from libcore import *
import time

#n = 1024
n = 64
img_tensor = torch.randn(1, n, n)

_, H, W = img_tensor.shape

data = img_tensor[None]
# show image
imshow_tensor(data)


############ Fourier  <-- frequency domain ########################
filter_np = np.array([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ])


h = np.zeros((H, W))
h[-2:,-2:] = filter_np[:2,:2]
h[-2:,:3]  = filter_np[:2,2:5]
h[:3,-2:]  = filter_np[2:5,:2]
h[:3,:3]   = filter_np[2:5,2:5]

H = h/np.sum(h)
t_start_fourier = time.time()
#result_fourier = np.real(ifft2(fft2(img_tensor.squeeze(0)) * fft2(H)))
result_fourier = np.real(ifft2(fft2(img_tensor.squeeze(0)) * fft2(filter_np, s=img_tensor.squeeze(0).shape)))
t_elapsed_fourier = time.time() - t_start_fourier
# show result of convolution us fourier
imshow_img(result_fourier)


######### nn.functional.conv2D ##########################
def conv2d(data, filters):
    return F.conv2d(data, filters, padding=2)

filter_ts = torch.tensor([
        [0.,.7,.8,.7,0.],
        [.7,1.,1.,1.,.7],
        [.8,1.,1.,1.,.8],
        [.7,1.,1.,1.,.7],
        [0.,.7,.8,.7,0.],
    ])
    
# [1,1,5,5]    
filter_tensor = filter_ts[None, None]
t_start_conv2d = time.time()
result_conv2d = conv2d(data, filter_tensor)
t_elapsed_conv2d = time.time() - t_start_conv2d
# show result of Convolution Operator
imshow_tensor(result_conv2d)


################ CPU nn.Conv2D ########################
model = torch.nn.Conv2d(1,1,5, padding=2, stride=1)
#print('weight: ', model.weight.shape)
model.weight = torch.nn.Parameter(filter_tensor.float(), requires_grad=False)
t_start_conv2d_cpu = time.time()
result_conv2d_cpu = model(data)
t_elapsed_conv2d_cpu = time.time() - t_start_conv2d_cpu
#print('data: ', model.weight)
imshow_tensor(result_conv2d_cpu.detach().cpu())

################ GPU nn.Conv2D ########################
device = "cuda:0"
model_gpu = torch.nn.Conv2d(1,1,5, padding=2, stride=1)
#print('weight: ', model.weight.shape)
model_gpu.weight = torch.nn.Parameter(filter_tensor.float(), requires_grad=False)
model_gpu.to(device)
data_gpu = data.to('cuda:0')
t_start_conv2d_gpu = time.time()
result_conv2d_gpu = model_gpu(data_gpu)
t_elapsed_conv2d_gpu = time.time() - t_start_conv2d_gpu
imshow_tensor(result_conv2d_gpu.detach().cpu())

#imshow_tensor(result_conv2d_gpu.detach().cpu())
print('functional: {}, cpu: {}, gpu: {}, fourier: {}'.format(t_elapsed_conv2d, t_elapsed_conv2d_cpu, t_elapsed_conv2d_gpu, t_elapsed_fourier))

# =============================================================================
# ratio_fourier = t_elapsed_conv2d_gpu/t_elapsed_fourier
# ratio_cpu = t_elapsed_conv2d_gpu/t_elapsed_conv2d_cpu
# print('gpu/fourier: {}, gpu/cpu: {}'.format(ratio_fourier, ratio_cpu))
# =============================================================================