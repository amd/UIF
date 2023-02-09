#  The MIT License (MIT)
#
#  Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the 'Software'), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""MIGraphX Imagenet classification example"""
import argparse
import cv2
import numpy as np
import migraphx

def get_arguments():
    """Parse argument list"""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--onnx_file',
                       help="name of imagenet ONNX model file")
    group.add_argument('--mxr_file',
                       help="name of imagenet Y-model file")
    parser.add_argument('--image', required=True,
                        help="name of input image")
    arguments = parser.parse_args()
    return arguments

def make_nxn(image, n):
    """Convert image to square size of nxn"""
    height, width, _ = image.shape
#    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if height > width:
        dif = height - width
        bar = dif // 2
        square = image[(bar + (dif % 2)):(height - bar), :]
        return cv2.resize(square, (n, n))
    elif width > height:
        dif = width - height
        bar = dif // 2
        square = image[:, (bar + (dif % 2)):(width - bar)]
        return cv2.resize(square, (n, n))
    else:
        return cv2.resize(image, (n, n))

def preprocess(img_data):
    """Normalize image and convert to standard size"""
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i, :, :] = (img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data

if __name__ == "__main__":
    args = get_arguments()

if args.onnx_file is not None:
    model = migraphx.parse_onnx(args.onnx_file)
    model.compile(migraphx.get_target("gpu"))
elif args.mxr_file is not None:
    model = migraphx.load(args.mxr_file)

# expect an input shape of batch=1,channels=3,height=N,width=N
input_name = model.get_parameter_names()[0]
input_shape = model.get_parameter_shapes()[input_name]
if (len(input_shape.lens()) != 4) \
   or (input_shape.lens()[0] != 1) \
   or (input_shape.lens()[2] != input_shape.lens()[3]):
    print("unexpected input shape for imagenet model: ", input_shape)
    quit()
else:
    input_size = input_shape.lens()[2]

img = cv2.imread(args.image, cv2.IMREAD_COLOR)

# process image
cropped = make_nxn(img, input_size)
chw = cropped.transpose(2, 0, 1)
prep = preprocess(chw)
data = np.expand_dims(prep.astype('float32'), 0)

# predict the output class
results = model.run({input_name : data})

print("Top bucket = ", np.argmax(np.array(results[0])))
