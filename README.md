# matcaffe2caffe
Convert a matcaffe model (column major) to a pycaffe or c++ caffe (row major) model.

In most cases you do not need to do this as you can just transpose the input image to get the same result.
However, in some cases, for example when you need to use [ncnn](https://github.com/Tencent/ncnn), the offical
script to convert a caffe model to ncnn model can only apply on a row major model, so you may need this tool
to convert the matcaffe model first. I write this tool when I need to convert the [MTCNN](https://github.com/kpzhang93/MTCNN_face_detection_alignment) model to a ncnn model. I do not do many tests so
if you meet any problem please open an issue.


# HowTo
```
$ python matcaffe2caffe.py --help
usage: matcaffe2caffe.py [-h] [--proto PROTO] [--model MODEL]
                           [--output OUTPUT]

convert a matcaffe model(column major) to a normal caffe model(row major)

optional arguments:
  -h, --help       show this help message and exit
  --proto PROTO    path to deploy prototxt.
  --model MODEL    path to pretrained weights
  --output OUTPUT  path to output model

$ python matcaffe2caffe.py --proto det1.prototxt --model det1.caffemodel --output det1_py.caffemodel
```

# Dependencies
pycaffe, numpy.
