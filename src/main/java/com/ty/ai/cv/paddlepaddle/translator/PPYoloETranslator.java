package com.ty.ai.cv.paddlepaddle.translator;

import ai.djl.modality.cv.Image;

/**
 * PPYoloE Translator
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
public class PPYoloETranslator extends PaddleDetectionTranslator{

    public PPYoloETranslator(int target_width, int target_height, boolean keep_ratio) {
        this(target_width, target_height, keep_ratio, 0.6f);
    }

    public PPYoloETranslator(int target_width, int target_height, boolean keep_ratio, float threshold) {
        this(target_width, target_height, keep_ratio, threshold, Image.Interpolation.BILINEAR);
    }

    public PPYoloETranslator(int target_width, int target_height, boolean keep_ratio, float threshold, Image.Interpolation interp) {
        super(target_width, target_height, keep_ratio, threshold, interp);
    }
}
