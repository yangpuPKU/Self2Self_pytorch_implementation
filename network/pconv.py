from tensor_type import Tensor4d, Tensor3d, Tensor
import math
from typing import Tuple, Union
import torch
from torch import nn

TupleInt = Union[int, Tuple[int, int]]


class PConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: TupleInt = 1,
        stride: TupleInt = 1,
        padding: TupleInt = 0,
        dilation: TupleInt = 1,
        bias: bool = False,
        legacy_behaviour: bool = False,
    ):
        """Partial Convolution on 2D input.
        :param in_channels:     see torch.nn.Conv2d
        :param out_channels:    see torch.nn.Conv2d
        :param kernel_size:     see torch.nn.Conv2d
        :param stride:          see torch.nn.Conv2d
        :param padding:         see torch.nn.Conv2d
        :param dilation:        see torch.nn.Conv2d
        :param bias:            see torch.nn.Conv2d
        :param legacy_behaviour: Tries to replicate Guilin's implementation's numerical error when handling the bias,
        but in doing so, it does extraneous operations that could be avoided and still result in *almost* the same
        result, at a tolerance of 0.00000458 % on the cuDNN 11.4 backend. Can safely be False for real life
        applications.
        """
        super().__init__()

        # Set this to True, and the output is guaranteed to be exactly the same as PConvGuilin and PConvRFR
        # Set this to False, and the output will be very very close, but with some numerical errors removed/added,
        # even though formally the maths are equivalent.
        self.legacy_behaviour = legacy_behaviour

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = self._to_int_tuple(kernel_size)
        self.stride = self._to_int_tuple(stride)
        self.padding = self._to_int_tuple(padding)
        self.dilation = self._to_int_tuple(dilation)
        self.use_bias = bias

        conv_kwargs = dict(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            bias=False,
        )

        # Don't use a bias here, we handle the bias manually to speed up computation
        self.regular_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs)

        # I found a way to avoid doing a in_channels --> out_channels conv and instead just do a
        # 1 channel in --> 1 channel out conv and then just scale the output of the conv by the number
        # of input channels, and repeat the resulting tensor to have "out channels"
        # This saves 1) a lot of memory because no need to pad before the conv
        #            2) a lot of computation because the convolution is way smaller (in_c * out_c times less operations)
        # It's also possible to avoid repeating the tensor to have "out channels", and instead use broadcasting
        # when doing operations. This further reduces the number of operations to do and is equivalent,
        # and especially the amount of memory used.
        # self.mask_conv = nn.Conv2d(in_channels=1, out_channels=1, **conv_kwargs)
        self.mask_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, **conv_kwargs)

        # Inits
        self.regular_conv.apply(
            lambda m: nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        )

        # the mask convolution should be a constant operation
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        for param in self.mask_conv.parameters():
            param.requires_grad = False

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(1, self.out_channels, 1, 1))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            # This is how nn._ConvNd initialises its weights
            nn.init.kaiming_uniform_(self.regular_conv.weight, a=math.sqrt(5))

            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.regular_conv.weight
                )
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias.view(self.out_channels), -bound, bound)

    def forward(self, x: Tensor4d, mask: Tensor4d) -> Tuple[Tensor4d, Tensor4d]:
        """Performs the 2D partial convolution.
        About the mask:
            - its dtype should be torch.float32
            - its values should be EITHER 0.0 OR 1.0, not in between
            - it should not have a channel dimensions. Just (batch, height, width).
        The returned mask is guaranteed to also match these criteria.
        This returns a tuple containing:
            - the result of the partial convolution on the input x.
            - the "updated mask", which is slightly "closed off". It is a "binary" mask of dtype float,
              containing values of either 0.0 or 1.0 (nothing in between).
        :param x: The input image batch, a 4d tensor of traditional batch, channel, height, width.
        :param mask: This takes as input a 3d binary (0.0 OR 1.0) mask of dtype=float
        :return: a tuple (output, updated_mask)
        """
        Tensor4d.check(x)
        batch, channels, h, w = x.shape
        Tensor[batch, channels, h, w].check(mask)

        if mask.dtype != torch.float32:
            raise TypeError(
                "mask should have dtype=torch.float32 with values being either 0.0 or 1.0"
            )

        if x.dtype != torch.float32:
            raise TypeError("x should have dtype=torch.float32")

        output = self.regular_conv(x * mask)
        _, _, conv_h, conv_w = output.shape

        update_mask: Tensor[batch, channels, conv_h, conv_w]
        mask_ratio: Tensor[batch, channels, conv_h, conv_w]
        with torch.no_grad():
            mask_ratio, update_mask = self.compute_masks(mask)

        if self.use_bias:
            if self.legacy_behaviour:
                # Doing this is entirely pointless. However, the legacy Guilin's implementation does it and
                # if I don't do it, I get a relative numerical error of about 0.00000458 %
                output += self.bias
                output -= self.bias

            output *= mask_ratio  # Multiply by the sum(1)/sum(mask) ratios
            output += self.bias  # Add the bias *after* mask_ratio, not before !
            output *= update_mask  # Nullify pixels outside the valid mask
        else:
            output *= mask_ratio

        return output, update_mask

    def compute_masks(self, mask: Tensor4d) -> Tuple[Tensor4d, Tensor4d]:
        """
        This computes two masks:
         - the update_mask is a binary mask that has 1 if the pixel was used in the convolution, and 0 otherwise
         - the mask_ratio which has value sum(1)/sum(mask) if the pixel was used in the convolution, and 0 otherwise
         * sum(1) means the sum of a kernel full of ones of equivalent size as the self.regular_conv's kernel.
           It is usually calculated as self.in_channels * self.kernel_size ** 2, assuming a square kernel.
         * sum(mask) means the sum of ones and zeros of the mask in a particular region.
           If the region is entirely valid, then sum(mask) = sum(1) but if the region is only partially within the mask,
           then 0 < sum(mask) < sum(1).
           sum(mask) is calculated specifically in the vicinity of the pixel, and is pixel dependant.
         * mask_ratio is Tensor4d with the channel dimension as a singleton, and is NOT binary.
           It has values between 0 and sum(1) (included).
         * update_mask is a Tensor4d with the channel dimension as a singleton, and is "binary" (either 0.0 or 1.0).
        :param mask: the input "binary" mask. It has to be a dtype=float32, but containing only values 0.0 or 1.0.
        :return: mask_ratio, update_mask
        """
        update_mask = self.mask_conv(mask)
        # Make values where update_mask==0 be super high
        # and otherwise computes the sum(ones)/sum(mask) value for other entries
        # noinspection PyTypeChecker
        mask_ratio = self.kernel_size[0] * self.kernel_size[1] / (update_mask + 1e-8)
        # Once we've normalised the values in update_mask and saved them elsewhere, we can now ignore their value
        # and return update_mask to a binary mask
        update_mask = torch.clamp(update_mask, 0, 1)
        # Then multiplies those super high values by zero so we cancel them out
        mask_ratio *= update_mask
        # We can discard the extra channel dimension what was just there to help with broadcasting

        return mask_ratio, update_mask

    @staticmethod
    def _to_int_tuple(v: TupleInt) -> Tuple[int, int]:
        if not isinstance(v, tuple):
            return v, v
        else:
            return v

    def set_weight(self, w):
        with torch.no_grad():
            self.regular_conv.weight.copy_(w)

        return self

    def set_bias(self, b):
        with torch.no_grad():
            self.bias.copy_(b.view(1, self.out_channels, 1, 1))

        return self

    def get_weight(self):
        return self.regular_conv.weight

    def get_bias(self):
        return self.bias