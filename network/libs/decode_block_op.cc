/*
* Project SketchCNN
*
*   Author: Changjian Li (chjili2011@gmail.com),
*   Copyright (c) 2018. All Rights Reserved.
*
* ==============================================================================
*/

/*
*   Custom decoder:
*       Given input date item, we first uncompress it and then divided them into input and label tensors.
*
*/


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "opencv2/opencv.hpp"
#include "zlib.h"

#include <complex>
#include <vector>

namespace tensorflow {

    const int32 input_channel = 6;
    const int32 output_channel = 17;

    REGISTER_OP("DecodeBlock")
            .Input("byte_stream: string")
            .Attr("tensor_size: list(int) >= 3")
            .Output("input_data: float")
            .Output("label_data: float")
            .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                std::vector<int32> t_size;
                TF_RETURN_IF_ERROR(c->GetAttr("tensor_size", &t_size));
                c->set_output(0, c->MakeShape({t_size[0], t_size[1], input_channel}));
                c->set_output(1, c->MakeShape({t_size[0], t_size[1], output_channel}));
                return Status::OK();
            })
            .Doc(R"doc(The decoder of multi-channel image data block)doc");

    class DecodeBlockOp : public OpKernel {
    public:
        explicit DecodeBlockOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("tensor_size", &this->tensor_size));
            OP_REQUIRES(context, this->tensor_size.size() == 3, errors::InvalidArgument("target tensor size must be 3-d, got ", this->tensor_size.size()));
        }

        void Compute(OpKernelContext* context) override {
            // Grab the input tensor
            const Tensor& contents = context->input(0);
            OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()), errors::InvalidArgument("DecodeBlock expect a scalar, got shape ",
                                                                                                       contents.shape().DebugString()));
            const StringPiece input_bytes = contents.scalar<string>()();

            // allocate the output tensor
            Tensor* input_data_tensor = nullptr;
            std::vector<int64> input_tensor_size;
            input_tensor_size.push_back(this->tensor_size[0]);
            input_tensor_size.push_back(this->tensor_size[1]);
            input_tensor_size.push_back(input_channel);
            TensorShape input_tensor_shape = TensorShape(gtl::ArraySlice<int64>{input_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("input_data", input_tensor_shape, &input_data_tensor));

            Tensor* label_data_tensor = nullptr;
            std::vector<int64> label_tensor_size;
            label_tensor_size.push_back(this->tensor_size[0]);
            label_tensor_size.push_back(this->tensor_size[1]);
            label_tensor_size.push_back(output_channel);
            TensorShape label_tensor_shape = TensorShape(gtl::ArraySlice<int64>{label_tensor_size});
            OP_REQUIRES_OK(context, context->allocate_output("label_data", label_tensor_shape, &label_data_tensor));

            // assemble data into tensor
            auto input_data_ptr = input_data_tensor->flat<float>();
            auto label_data_ptr = label_data_tensor->flat<float>();

            // uncompress the byte stream
            int out_data_size = -1;
            float* inflate_data = inflation_byte(input_bytes, out_data_size);
            OP_REQUIRES(context, out_data_size > 0, errors::InvalidArgument("Zlib inflation error, got size: ", out_data_size));
            OP_REQUIRES(context, (out_data_size - (int)(this->tensor_size[0]*this->tensor_size[1]*this->tensor_size[2])) == 0,
                        errors::InvalidArgument("Inflated data mismatch, got ", out_data_size));

            // set tensor value
            int64 height = this->tensor_size[0];
            int64 width = this->tensor_size[1];
            int64 channel = this->tensor_size[2];

            for(int ritr=0; ritr<height; ritr++)
            {
                for(int citr=0; citr<width; citr++)
                {
                    int64 idx = ritr*width + citr;

                    input_data_ptr(idx*input_channel+0) = inflate_data[idx*channel+0];      // npr lines
                    input_data_ptr(idx*input_channel+3) = inflate_data[idx*channel+6];      // depth sample
                    input_data_ptr(idx*input_channel+1) = inflate_data[idx*channel+11];     // contour df
                    input_data_ptr(idx*input_channel+2) = inflate_data[idx*channel+12];     // line df
                    input_data_ptr(idx*input_channel+4) = inflate_data[idx*channel+17];     // feature line mask
                    if(inflate_data[idx*channel+17] > 0.0)                                  // feature line mask inverse
                        input_data_ptr(idx*input_channel+5) = 0.0;
                    else
                        input_data_ptr(idx*input_channel+5) = 1.0;

                    label_data_ptr(idx*output_channel+0) = (inflate_data[idx*channel+8] + 1.0) / 2.0;       // gt normal x
                    label_data_ptr(idx*output_channel+1) = (inflate_data[idx*channel+9] + 1.0) / 2.0;       // gt normal y
                    label_data_ptr(idx*output_channel+2) = (inflate_data[idx*channel+10] + 1.0) / 2.0;      // gt normal z
                    label_data_ptr(idx*output_channel+3) = inflate_data[idx*channel+7];                     // gt depth
                    label_data_ptr(idx*output_channel+4) = inflate_data[idx*channel+2];                     // gt a_real
                    label_data_ptr(idx*output_channel+5) = inflate_data[idx*channel+3];                     // gt a_img
                    label_data_ptr(idx*output_channel+6) = inflate_data[idx*channel+4];                     // gt b_real
                    label_data_ptr(idx*output_channel+7) = inflate_data[idx*channel+5];                     // gt b_img
                    label_data_ptr(idx*output_channel+8) = inflate_data[idx*channel+1];                     // shape mask

                    if(inflate_data[idx*channel+6] > 0.0)                                                   // depth sample mask
                        label_data_ptr(idx*output_channel+9) = 1.0;
                    else
                        label_data_ptr(idx*output_channel+9) = 0.0;

                    if(inflate_data[idx*channel+15] > 0)                                                    // contour mask inv
                        label_data_ptr(idx*output_channel+10) = 0.0;
                    else
                        label_data_ptr(idx*output_channel+10) = 1.0;

                    if(inflate_data[idx*channel+0] < 1.0)                                                   // line mask inv
                    {
                        label_data_ptr(idx*output_channel+12) = 0.0;
                        label_data_ptr(idx*output_channel+11) = 1.0;
                    }
                    else
                    {
                        label_data_ptr(idx*output_channel+12) = 1.0;
                        label_data_ptr(idx*output_channel+11) = 0.0;
                    }

                    label_data_ptr(idx*output_channel+13) = inflate_data[idx*channel+14];                   // 2d mask

                    // weight map (f(x) = exp(-20.0*x)+1.0)
                    if(inflate_data[idx*channel+1] > 0.0)
                        label_data_ptr(idx*output_channel+14) = std::exp(-20.0*inflate_data[idx*channel+12])+1.0;  // weight map
                    else
                        label_data_ptr(idx*output_channel+14) = 0.0;

                    label_data_ptr(idx*output_channel+15) = inflate_data[idx*channel+18];                   // selected line mask

                    label_data_ptr(idx*output_channel+16) = inflate_data[idx*channel+24];                   // \deltaN_V*V, scalar
                }
            }

            delete[] inflate_data;
        }

    private:
        float* inflation_byte(const StringPiece &input_bytes, int& out_size)
        {
            // zipper stream
            z_stream infstream;
            infstream.zalloc = Z_NULL;
            infstream.zfree = Z_NULL;
            infstream.opaque = Z_NULL;

            // set input, output
            Byte* uncompressed_data = new Byte[100000000];

            infstream.avail_in = (uInt)input_bytes.size();
            infstream.next_in = (Bytef*)input_bytes.data();
            infstream.avail_out = (uLong)100000000;
            infstream.next_out = uncompressed_data;

            // uncompress work
            int nErr, real_out_size = -1;

            nErr = inflateInit(&infstream);
            if(nErr != Z_OK)
            {
                out_size = -1;
                return nullptr;
            }
            nErr = inflate(&infstream, Z_FINISH);
            if(nErr == Z_STREAM_END)
            {
                real_out_size = (int)infstream.total_out;
            }
            inflateEnd(&infstream);

            // assign data
            real_out_size /= 4;
            out_size = real_out_size;

            return (float *)uncompressed_data;
        }


    private:
        std::vector<int64> tensor_size;
    };

    REGISTER_KERNEL_BUILDER(Name("DecodeBlock").Device(DEVICE_CPU), DecodeBlockOp);

}
