import tvm
import math
import numpy as np
from tvm import te
from tvm import topi
from tvm import tir
from functools import reduce

#################### helper functions - custom reducer ####################


def fidentity_sum(t0, t1):
    """neutral operation for the reducer"""
    return tvm.tir.const(0, dtype=t0), tvm.tir.const(0, dtype=t1)


def fcombine_sum(acc, value):  # x = reduced value, y = current value, 0 = mean, 1 = var
    """custom reduction for sum"""
    mean = acc[0] + value[0]
    var = acc[1] + value[1]
    return mean, var


def fidentity_max(t0, t1):
    """neutral operation for the reducer"""
    # -inf for mean as absolute minimum and 0 for var
    return -tvm.tir.max_value(dtype=t0), tvm.tir.const(0, dtype=t1)

def fcombine_max(acc, value): 
    """custom reduction for max"""
    mean, var = max_pfp_marvin(acc[0], acc[1], value[0], value[1])
    return mean, var


my_sum_combined = te.comm_reducer(
    fcombine_sum, fidentity_sum, name="my_sum_combined")
my_max_combined = te.comm_reducer( # TODO offer more options for max calc, see thesis
    fcombine_max, fidentity_max, name="my_max_combined")


# inspired by d2ltvm package
# pad innermost 2 dimensions with given value
def pad_array(X,padding, val=0):
    """Pad X with the given value in 2-D

    padding : height and width padding
    val : padding value, default 0
    """
    ph, pw = padding[0], padding[1]

    assert len(X.shape) >= 2
    nh, nw = X.shape[-2], X.shape[-1]
    return te.compute(
            (*X.shape[0:-2], nh+ph*2, nw+pw*2),
            lambda *i: te.if_then_else(
                te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
                val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
            name='padded_' + X.name)


#################### DENSE ####################

def dense_pfp(x_m: te.Tensor, x_s: te.Tensor, w_m: te.Tensor, w_s: te.Tensor, bias_mean: te.Tensor = None, bias_var: te.Tensor = None, convert_var_activations_to_2ndrawmoment: bool = False, convert_var_weights_to_2ndrawmoment: bool = False, packed: bool = False, bn: int = 32) -> te.Tensor:
    """# pfp-dense operation
    x_m: input data mu (mean) (shape MxN)
    x_s: input data sigma (raw 2nd moment)
    w_m: weight mu (mean) (shape KxN [this is transposed])
    w_s: weight sigma (raw 2nd moment)
    bias_mean: mean of biases
    bias_var:  variance of bias
    variance_mode_input: input sigma is treated as variance and converted to second raw moment
    variance_mode_weights: weight sigmas are treated as variances and converted to second raw moment
    packed: packed access to X tensor
    bn: blocking size for packing
    returns mean and variance of x*wT (MxK)
    """

    M = x_m.shape[0]
    N = x_m.shape[1]
    K = w_m.shape[0]

    out_shape = (M, K)
    k = te.reduce_axis((0, N), "k")

    if convert_var_activations_to_2ndrawmoment:
        # variance --> 2nd raw moment  
        x_s = x_s + x_m * x_m

    if convert_var_weights_to_2ndrawmoment:
        # variance --> 2nd raw moment
        w_s = w_s + w_m * w_m
    
    # attention x_s and w_s are 2nd raw moments: E[x^2] = mu_x^2 + sigma_x^2 and acordingly for w. 


    if packed:
        print("Warning: 'packed'-dense is untested!")
        packed_w_m = te.compute(
            (N // bn, K, bn), lambda bigN, k, littleN: w_m[k, bigN * bn + littleN], name="packedWM"
        )
        packed_w_s = te.compute(
            (N // bn, K, bn), lambda bigN, k, littleN: w_s[k, bigN * bn + littleN], name="packedWS"
        )
        if x_s is not None:
            a_m, a_s = te.compute(
                out_shape,
                lambda i, j: my_sum_combined(
                    (
                        m := x_m[i, k] * packed_w_m[k // bn, j, tvm.tir.indexmod(k, bn)],
                        x_s[i, k] * packed_w_s[k // bn, j, tvm.tir.indexmod(k, bn)] - m * m,
                    ),
                    axis=k,
                ),
                name="dense_pfp_packed",
            )
        else: # x_s --> None in first layer # attention w_var is needed not 2ndraw
            a_m, a_s = te.compute(
                out_shape,
                lambda i, j: my_sum_combined(
                    (
                        x_m[i, k] * packed_w_m[k // bn, j, tvm.tir.indexmod(k, bn)],
                        x_m[i, k] * x_m[i, k] * packed_w_s[k // bn, j, tvm.tir.indexmod(k, bn)],
                    ),
                    axis=k,
                ),
                name="dense_pfp_packed_first_layer",
            )
    else: # non-packed
        out_shape = (x_m.shape[0], w_m.shape[0])
        k = te.reduce_axis((0, x_m.shape[1]), "k")
        if x_s is not None:
            a_m, a_s = te.compute(
                out_shape,
                lambda i, j: my_sum_combined(
                    (
                        m := x_m[i, k] * w_m[j, k],
                        x_s[i, k] * w_s[j, k] - m * m
                    ),
                    axis=k,
                ),
                name="dense_pfp",
            )
        else: # x_s is None in first layer # attention w_s is var not 2nd raw moment.
            a_m, a_s = te.compute(
                out_shape,
                lambda i, j: my_sum_combined(
                    (
                        x_m[i, k] * w_m[j, k],
                        x_m[i, k] * x_m[i, k] * w_s[j, k],
                    ),
                    axis=k,
                ),
                name="dense_pfp_first_layer",
            )

    if bias_mean is not None or bias_var is not None:
        a_m, a_s = te.compute(out_shape, 
                              lambda i, j: (
                                  a_m[i, j] + (bias_mean[j] if bias_mean is not None else 0), 
                                  a_s[i, j] + (bias_var[j] if bias_var is not None else 0)), 
                            name="pfp_add_bias")

    return a_m, a_s


#################### ACTIVATION FUNCTIONS ####################


def sigmoid_pfp(a_m: te.Tensor, a_s: te.Tensor, variance_mode: bool = False) -> te.Tensor:
    """pfp-Sigmoid operation 
    a_m: activation mu (mean)
    a_s: activation sigma (variance)
    to_variance: returns variance instead of second moment
    returns mean and second moment for the next layer
    """

    alpha = 1.1715728752538097  # 4-2sqrt(2)
    beta = 0.8813735870195428  # -log(sqrt(2)-1)
    gamma = 0.6266570686577501  # sqrt(pi/8)
    a_s = topi.nn.relu(a_s) # a_s must not be negative
    x_m, x_s = topi.sigmoid(a_m/topi.sqrt(1+gamma**2*a_s)), topi.sigmoid(
        (alpha*(a_m-beta))/topi.sqrt(1+alpha**2*gamma**2*a_s))
    
    if variance_mode:
        x_s = x_s - (x_m * x_m)



    return x_m, x_s


def relu_pfp(a_m: te.Tensor, a_s: te.Tensor, variance_mode: bool = False) -> te.Tensor:
    """pfp-ReLU operation 
    a_m: activation mu (mean)
    a_s: activation sigma (variance)
    returns mean and second moment for the next layer
    """
    a_s = topi.nn.relu(a_s)# guard against negative numbers, a_s must not be negative
    sqrt1 = tvm.topi.sqrt(2*a_s)
    sqrt2 = tvm.topi.sqrt(a_s/(2*math.pi))

    erf = tvm.topi.erf(a_m/sqrt1)
    exp = tvm.topi.exp(-a_m*a_m/(2*a_s))

    x_m, x_s = te.compute(
        a_m.shape,
        lambda *i: (
            (a_m(*i))/2 * (1+erf(*i)) + sqrt2(*i)*exp(*i),
            (a_m(*i)*a_m(*i) + a_s(*i))/2 *
            (1+erf(*i)) + a_m(*i) * sqrt2(*i)*exp(*i)
        ),
        name="relu_pfp"
    )

    if variance_mode:
        x_s = x_s - (x_m * x_m)
    return x_m, x_s

def relu_pfp2(a_m: te.Tensor, a_s: te.Tensor, variance_mode: bool = False) -> te.Tensor:
    """pfp-ReLU operation 
    a_m: activation mu (mean)
    a_s: activation sigma (variance)
    returns mean and second moment for the next layer

    NOTE: This version computes verything in one loop, but it is half as fast as relu_pfp! so dont use it
    """
    a_s = topi.nn.relu(a_s)# guard against negative numbers, a_s must not be negative
   
    x_m, x_s = te.compute(
        a_m.shape,
        lambda *i: (
            (a_m(*i))/2 * (1+tir.erf(a_m(*i)/tir.sqrt(2*a_s(*i)))) + tir.sqrt(a_s(*i)/(2*math.pi))*tir.exp(-a_m(*i)*a_m(*i)/2*a_s(*i)),
            (a_m(*i)*a_m(*i) + a_s(*i))/2 * (1+tir.erf(a_m(*i)/tir.sqrt(2*a_s(*i)))) + a_m(*i) * tir.sqrt(a_s(*i)/(2*math.pi))*tir.exp(-a_m(*i)*a_m(*i)/2*a_s(*i))
        ),
        name="relu_pfp"
    )

    if variance_mode:
        x_s = x_s - (x_m * x_m)
    return x_m, x_s


#################### CONVOLUTION ####################

def conv_pfp_2d(input_mean: te.Tensor, input_var: te.Tensor, filter_mean: te.Tensor, filter_var: te.Tensor, stride: int, pad: int, bias_mean: te.Tensor, bias_var: te.Tensor, convert_var_activations_to_2ndrawmoment: bool = False, convert_var_weights_to_2ndrawmoment: bool = False, ceil_mode = True, dtype: str = "float32", layout: str = "NCHW") -> te.Tensor:
    if layout not in ["NCHW"]:#, "NHWC", "HWCN"]:
        return ValueError("Invalid layout in conv_pfp_2d: " + layout)

    # first layer mode without variance, thus 2nd raw momment = input_mean^2
    if convert_var_activations_to_2ndrawmoment:
        # variance --> 2nd Raw moment
        input_var = input_var + (input_mean * input_mean)

    if convert_var_weights_to_2ndrawmoment:
        # variance --> 2nd Raw moment
        filter_var = filter_var + (filter_mean * filter_mean)
        #pass

    N = input_mean.shape[layout.find("N")]
    C = input_mean.shape[layout.find("C")]
    H = input_mean.shape[layout.find("H")]
    W = input_mean.shape[layout.find("W")]

    if layout == "NCHW":
        out_layout = "OIHW"
    else:
        raise ValueError("Output layout not implemented for " + layout + " in conv_pfp_2d")

    kernel_size = filter_mean.shape[out_layout.find("W")] # or "H", should be identical for square filter  

    # PADDING
    padded_shape_list = [0, 0, 0, 0]
    padded_shape_list[layout.find("N")] = N
    padded_shape_list[layout.find("C")] = C
    padded_shape_list[layout.find("H")] = H + 2 * pad
    padded_shape_list[layout.find("W")] = W + 2 * pad
    padded_shape = tuple(i for i in padded_shape_list)

    if layout == "NCHW":
        padded_mean = te.compute(
            padded_shape,  
            lambda nn, cc, i, j: tvm.tir.if_then_else(
                tvm.tir.all(j >= pad, j - pad < H, i >= pad, i - pad < W),
                input_mean[nn, cc, i - pad, j - pad],
                tvm.tir.const(0.0, dtype),
            ),
            name="padded_mean",
        )
        padded_var = te.compute(
            padded_shape,  
            lambda nn, cc, i, j: tvm.tir.if_then_else(
                tvm.tir.all(j >= pad, j - pad < H, i >= pad, i - pad < W),
                input_var[nn, cc, i - pad, j - pad],
                tvm.tir.const(0.0, dtype),
            ),
            name="padded_var",
        ) if input_var is not None else None
    else:
        raise NotImplementedError("Padding not implemented for " + layout + " in conv_pfp_2d")

    
    #OUTPUT SIZE CALCULATIONS
    out_shape_list = [0, 0, 0, 0]
    out_channel = filter_mean.shape[out_layout.find("O")]
    out_batch = N

    divide = tvm.tir.ceildiv if ceil_mode else tvm.tir.floordiv
    out_width = divide((W - kernel_size + 2 * pad) , stride) + 1
    out_height = divide((H - kernel_size + 2 * pad) , stride) + 1


    out_shape_list[out_layout.find("O")] = out_batch
    out_shape_list[out_layout.find("I")] = out_channel
    out_shape_list[out_layout.find("H")] = out_height
    out_shape_list[out_layout.find("W")] = out_width

    out_shape = tuple(i for i in out_shape_list)
    print("inshape", input_mean.shape)
    print("outshape", out_shape)

    # Create reduction variables
    rc = te.reduce_axis((0, C), name="rc") # reduction over input channels
    ry = te.reduce_axis((0, kernel_size), name="ry") # reduction over Height
    rx = te.reduce_axis((0, kernel_size), name="rx") # reduction over Width
    # Compute the convolution
    if input_var is not None:
        a_m, a_s = te.compute(
            out_shape,
            lambda nn, oc, i, j: my_sum_combined((
                
                m := padded_mean[nn, rc, i * stride + ry, j * stride + rx] * filter_mean[oc, rc, ry, rx],
                padded_var[nn, rc, i * stride + ry, j * stride + rx] * filter_var[oc, rc, ry, rx] - m * m
                )  
                , axis=[rc, ry, rx]
            ),
            name="conv_pfp_2d",          # req. 2nd raw moments
        )
    else:
        a_m, a_s = te.compute(
            out_shape,
            lambda nn, oc, i, j: my_sum_combined((
                
                padded_mean[nn, rc, i * stride + ry, j * stride + rx] * filter_mean[oc, rc, ry, rx],
                padded_mean[nn, rc, i * stride + ry, j * stride + rx] * padded_mean[nn, rc, i * stride + ry, j * stride + rx] * filter_var[oc, rc, ry, rx]
                )  
                , axis=[rc, ry, rx]
            ),
            name="conv_pfp_2d_fist_layer",  # req. variances
        )

    if bias_mean is not None or bias_var is not None:
        a_m = te.compute(out_shape, 
                lambda nn, cc, i, j: a_m[nn, cc, i, j] + (bias_mean[cc] if bias_mean is not None else tvm.tir.const(0, dtype=dtype)), 
                name="conv_pfp_add_bias_m")
        a_s = te.compute(out_shape, 
                lambda nn, cc, i, j: a_s[nn, cc, i, j] + (bias_var[cc] if bias_var is not None else tvm.tir.const(0, dtype=dtype)), 
                name="conv_pfp_add_bias_s")
 
    return a_m, a_s

"""

            m := padded_mean[yy * stride + ry, xx * stride + rx, rc, nn] * filter_mean[ry, rx, rc, ff],
            -m * m + padded_var[yy * stride + ry, xx * stride + rx, rc, nn] * filter_var[ry, rx, rc, ff])  
            , axis=[ry, rx, rc]

"""

# MISCELANEOUS
# TODO dtype should be set from outside!
dtype = "float32"
zero = tvm.tir.const(0, dtype=dtype)
one = tvm.tir.const(1, dtype=dtype)
two = tvm.tir.const(2, dtype=dtype)
pi = tvm.tir.const(math.pi, dtype=dtype)

def pdf(x, mu=zero, sigma=one):  # pdf of standard normal, mu = 0, sigma =
    return te.div(one, sigma * te.sqrt(two*pi)) * te.exp(te.div(( te.power(te.div(x-mu, sigma),two) ), -two))


def cdf(x, mu=zero, sigma=one):  # cdf of standard normal
    return te.div((one + te.erf(te.div(x-mu, sigma * te.sqrt(two)))), two)


def max_pfp(mean1, var1, mean2, var2, epsilon=1e-4):
    # implementation according to thesis
    alpha = te.sqrt(var1 + var2 + epsilon)
    beta = te.div(mean1 - mean2, alpha)

    mean_out = mean1 * cdf(beta) + mean2 * cdf(-beta) + alpha * pdf(beta)
    var_out = (var1 + mean1) * cdf(beta) + (var2 + mean2) * pdf(-beta) + epsilon

    return mean_out, var_out

def max_pfp_marvin(mean1, var1, mean2, var2, epsilon=1e-4):
    # implementation according to thesis
    # var1 and var2 are variances here!
    a = te.sqrt(var1 + var2 + epsilon)
    alpha = te.div(mean1 - mean2, a)
    aux_erf = te.erf(alpha * (0.5 ** 0.5))
    cdf_alpha_pos = 0.5 * (1.0 + aux_erf)
    cdf_alpha_neg = 0.5 * (1.0 - aux_erf)
    pdf_norm = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_alpha = pdf_norm * te.exp(-0.5 * alpha * alpha)
    a_times_pdf_alpha = a * pdf_alpha
    mean_out = mean1 * cdf_alpha_pos + mean2 * cdf_alpha_neg + a_times_pdf_alpha
    var_out = (var1 + mean1* mean1) * cdf_alpha_pos \
          + (var2 + mean2*mean2) * cdf_alpha_neg \
          + (mean1 + mean2) * a_times_pdf_alpha \
          - mean_out*mean_out + epsilon
    var_out = topi.maximum(var_out, 0)

    return mean_out, var_out

"""
    # Gaussian approximation of the maximum of two Gaussians
    # Implementation according to:
    # Sinha et al.; Advances in Computation of the Maximum of a Set of Random Variables
   x a_sqr = v1 + v2 + epsilon
   x a = np.sqrt(a_sqr)
   x alpha = (m1 - m2) / a
   x
   x aux_erf = scipy.special.erf(alpha * (0.5 ** 0.5))
    cdf_alpha_pos = 0.5 * (1.0 + aux_erf)
    cdf_alpha_neg = 0.5 * (1.0 - aux_erf)
    pdf_norm = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_alpha = pdf_norm * np.exp(-0.5 * alpha**2)
    a_times_pdf_alpha = a * pdf_alpha

    m_max = m1 * cdf_alpha_pos + m2 * cdf_alpha_neg + a_times_pdf_alpha
    v_max = (v1 + m1**2) * cdf_alpha_pos \
          + (v2 + m2**2) * cdf_alpha_neg \
          + (m1 + m2) * a_times_pdf_alpha \
          - m_max**2 + epsilon

    return m_max, v_max

"""


def max_pool_pfp(data_mean, data_var,
                 pool_size=(1, 1),
                 stride=(1, 1),
                 dilation=(1, 1),  # not implemented
                 padding=(0, 0),
                 layout="NCHW",  # only NCHW for the moment
                 ceil_mode=False,
                 dtype = "float32"):
    # the used Max Pool algorithm uses mean and variances, not 2nd raw momemnt

    N = data_mean.shape[0]
    C = data_mean.shape[1]
    H = data_mean.shape[2]
    W = data_mean.shape[3]
    
    op = tvm.tir.ceildiv if ceil_mode else tvm.tir.floordiv
    H_out = op((H + 2 * padding[0] - dilation[0] * (pool_size[0] - 1) - 1) , stride[0]) + 1
    W_out = op((W + 2 * padding[1] - dilation[1] * (pool_size[1] - 1) - 1) , stride[1]) + 1


    out_shape = (N, C, H_out, W_out)
    kH = te.reduce_axis((0, pool_size[0]), "kH")
    kW = te.reduce_axis((0, pool_size[1]), "kW")

    data_mean_padded = pad_array(data_mean, padding, val=te.min_value(dtype=dtype)) if padding != (0,0) else data_mean
    data_var_padded = pad_array(data_var, padding, val=tvm.tir.const(0, dtype=dtype)) if padding != (0,0) else data_var

    mean_comp, var_comp = te.compute( # page 98 thesis
        out_shape, 
        lambda n, c, h, w: my_max_combined(
            (
                data_mean_padded[n, c, stride[0] * h + kW,stride[1]*w+kH], data_var_padded[n, c, stride[0] * h + kW,stride[1]*w+kH] 
            ),
            axis=[kH, kW],
        ),
        "max_pool_pfp")

    return mean_comp, var_comp

def lenet_pool(data_mean, data_var):

    stride=2
    epsilon = 1e-4 # 1e-6 default but NaNs occur, 1e-5 seems to work without NaNs

    N = data_mean.shape[0]
    C = data_mean.shape[1]
    H = data_mean.shape[2]
    W = data_mean.shape[3]

    
    H_out = tvm.tir.floordiv(H, stride) 
    W_out = tvm.tir.floordiv(W, stride) 


    out_shape_h = (N, C, H, W_out)

    mean_h, mean_h1 = te.compute( 
        out_shape_h, 
        lambda n, c, h, w: (
                data_mean[n, c, h, w* stride], data_mean[n, c, h, w*stride+1] 
        ),
        "lenet_pool_hm")
    var_h, var_h1 = te.compute( 
        out_shape_h, 
        lambda n, c, h, w: (
                data_var[n, c, h, w* stride], data_var[n, c, h, w*stride+1] 
        ),
        "lenet_pool_hv")

    a_h = topi.sqrt(var_h + var_h1 + epsilon)
    alpha_h = (mean_h - mean_h1)/ a_h
    aux_erf_h = topi.erf(alpha_h * (0.5 ** 0.5))
    cdf_alpha_pos_h = 0.5 * (1.0 + aux_erf_h)
    cdf_alpha_neg_h = 0.5 * (1.0 - aux_erf_h)
    pdf_norm_h = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_alpha_h = pdf_norm_h * topi.exp(-0.5 * alpha_h * alpha_h)
    a_times_pdf_alpha_h = a_h * pdf_alpha_h

    mean_out = mean_h * cdf_alpha_pos_h + mean_h1 * cdf_alpha_neg_h + a_times_pdf_alpha_h
    var_out = (var_h + mean_h* mean_h) * cdf_alpha_pos_h \
          + (var_h1 + mean_h1*mean_h1) * cdf_alpha_neg_h \
          + (mean_h + mean_h1) * a_times_pdf_alpha_h \
          - mean_out*mean_out + epsilon

    var_out = topi.maximum(var_out, 0)

    # TODO calculate the pooling in horizontal direction
    out_shape_v = (N, C, H_out, W_out)
    mean_v, mean_v1 = te.compute( 
        out_shape_v, 
        lambda n, c, h, w: (
            (
                mean_out[n, c, h*stride, w], mean_out[n, c, h*stride+1, w] 
            )
        ),
        "lenet_pool_vm")
    var_v, var_v1 = te.compute( 
        out_shape_v, 
        lambda n, c, h, w: (
            (
                var_out[n, c, h*stride, w], var_out[n, c, h*stride+1, w] 
            )
        ),
        "lenet_pool_vv")

    a_v = topi.sqrt(var_v + var_v1 + epsilon)
    alpha_v = (mean_v - mean_v1)/ a_v
    aux_erf_v = topi.erf(alpha_v * (0.5 ** 0.5))
    cdf_alpha_pos_v = 0.5 * (1.0 + aux_erf_v)
    cdf_alpha_neg_v = 0.5 * (1.0 - aux_erf_v)
    pdf_norm_v = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_alpha_v = pdf_norm_v * topi.exp(-0.5 * alpha_v * alpha_v)
    a_times_pdf_alpha_v = a_v * pdf_alpha_v

    mean_comp = mean_v * cdf_alpha_pos_v + mean_v1 * cdf_alpha_neg_v + a_times_pdf_alpha_v
    var_comp = (var_v + mean_v* mean_v) * cdf_alpha_pos_v \
          + (var_v1 + mean_v1*mean_v1) * cdf_alpha_neg_v \
          + (mean_v + mean_v1) * a_times_pdf_alpha_v \
          - mean_comp*mean_comp + epsilon

    var_comp = topi.maximum(var_comp, 0)

    return mean_comp, var_comp

def lenet_split(mean, var):

    N = mean.shape[0]
    C = mean.shape[1]
    H = mean.shape[2]
    W = mean.shape[3]
    out_shape = (N,C,H//2,W//2)

    m00, m01, m10, m11 = te.compute(
        out_shape,
        lambda n, c, h, w: (
            mean[n, c, h, w], mean[n, c, h, 14+w], mean[n, c, 14+h, w], mean[n, c, 14+h, 14+w]
        ),
        "lenet_split_m"   
    )

    v00, v01, v10, v11 = te.compute(
        out_shape,
        lambda n, c, h, w: (
            var[n, c, h, w], var[n, c, h, 14+w], var[n, c, 14+h, w], var[n, c, 14+h, 14+w]
        ),
        "lenet_split_v"   
    )

    return m00, v00, m01, v01, m10, v10, m11, v11


# currently no plan to implement this
def global_average_pfp():
    raise NotImplementedError

def batch_norm_pfp(data_mean, data_var, gamma, beta, epsilon=1e-5):
    # assert data_mean.shape == data_var.shape
    axis = 1
    # assuming NCHW

    shape = [1] * len(data_mean.shape) # shape = [1, 1, 1, 1]
    shape[axis] = data_mean.shape[axis]# shape = [1, axis, 1, 1]
    reduce_axes = list(range(len(data_mean.shape)))
    reduce_axes.remove(axis)
    shape_prod = reduce(lambda x, y: x * y, [data_mean.shape[ax] for ax in reduce_axes], 1)

    mu_bn = topi.sum(data_mean, axis=reduce_axes) / shape_prod # 5.10 left
    mu_bn_rs = topi.reshape(mu_bn, shape)

    sigma_bn = (
        topi.sum(data_var + (data_mean - mu_bn_rs) * (data_mean - mu_bn_rs), axis=reduce_axes) / shape_prod # 5.10 right, TODO shape_prod N-1 at axis instead of N
    )
    sigma_bn_rs = topi.reshape(sigma_bn, shape)

    data_out = (data_mean - mu_bn_rs) / (topi.math.sqrt(sigma_bn_rs + epsilon)) * gamma + beta # 5.11 left
    
    
    var_out = data_var / (sigma_bn_rs + epsilon) * gamma * gamma # 5.11 right

    return data_out, var_out


def batch_norm_pfp_inference(data_mean, data_var, moving_mean, moving_var, alpha=1e-5):
    """
    untested, may contain false logic !!!! FIXME
    test inference should only use moving averages, but there is no reference code 
    """
    axis = 1
    reduce_axes = list(range(len(data_mean.shape)))
    reduce_axes.remove(axis)
    shape_prod = reduce(lambda x, y: x * y, [data_mean.shape[ax] for ax in reduce_axes], 1)

    mu_bn = topi.sum(data_mean, axis=reduce_axes) / shape_prod # 5.10 left
    mu_bn_rs = topi.reshape(mu_bn, shape)

    sigma_bn = (
        topi.sum(data_var + (data_mean - mu_bn_rs) * (data_mean - mu_bn_rs), axis=reduce_axes) / shape_prod # 5.10 right, TODO shape_prod N-1 at axis instead of N
    )
    sigma_bn_rs = topi.reshape(sigma_bn, shape)

    shape = [1] * len(data_mean.shape)
    shape[axis] = data_mean.shape[axis]

    moving_mean_rs = topi.reshape(moving_mean, shape)
    moving_var_rs = topi.reshape(moving_var, shape)
    
    out_mean = alpha * mu_bn_rs + (1 - alpha) * moving_mean_rs
    out_var = alpha * sigma_bn_rs + (1 - alpha) * moving_var_rs
    
    # comment from original batch_norm:
    # Moving mean and var aren't updated during test. To avoid
    # placeholder reuse, we multiply by 1 and return them.
    return [out_mean, out_var, moving_mean * 1, moving_var * 1]
