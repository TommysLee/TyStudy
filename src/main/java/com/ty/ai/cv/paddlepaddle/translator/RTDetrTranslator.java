package com.ty.ai.cv.paddlepaddle.translator;

import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.TranslatorContext;
import lombok.extern.slf4j.Slf4j;

/**
 * RT-DETR Translator
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
@Slf4j
public class RTDetrTranslator extends PaddleDetectionTranslator{

    public RTDetrTranslator(int target_width, int target_height, boolean keep_ratio) {
        this(target_width, target_height, keep_ratio, 0.6f);
    }

    public RTDetrTranslator(int target_width, int target_height, boolean keep_ratio, float threshold) {
        this(target_width, target_height, keep_ratio, threshold, Image.Interpolation.BILINEAR);
    }

    public RTDetrTranslator(int target_width, int target_height, boolean keep_ratio, float threshold, Image.Interpolation interp) {
        super(target_width, target_height, keep_ratio, threshold, interp);
    }

    /**
     * Processes the input and converts it to NDList.
     *
     * @param ctx   the toolkit for creating the input NDArray
     * @param input the input object
     * @return the {@link NDList} after pre-processing
     * @throws Exception if an error occurs during processing input
     */
    @Override
    public NDList processInput(TranslatorContext ctx, Image input) throws Exception {
        NDList ndList = super.processInput(ctx, input);

        // RT-DETR 模型前置处理 需要构造三个参数：[image_shape, image, scale_factor]
        NDArray im_shape = ctx.getNDManager().create(this.target_size);
        im_shape = im_shape.toType(DataType.FLOAT32, false).expandDims(0);

        ndList.add(0, im_shape);
        return ndList;
    }
}
