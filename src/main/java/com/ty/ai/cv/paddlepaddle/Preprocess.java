package com.ty.ai.cv.paddlepaddle;

import ai.djl.modality.cv.Image.Interpolation;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.google.common.collect.Lists;

import java.util.List;

/**
 * 封装常用的数据预处理算子
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
public final class Preprocess {

    /**
     * Resize Image by target_size
     *
     * @param im            image ndarray
     * @param target_size   the target size of image
     * @param keep_ratio    whether keep_ratio or not
     * @param interp        interpolation method of resize
     * @param manager       NDArray Manager
     * @return List<NDArray>
     */
    public static List<NDArray> resize(NDArray im, int[] target_size, boolean keep_ratio, Interpolation interp, NDManager manager) {
        // Image Shape HWC
        Shape origin_shape = im.getShape();

        // 原始图片高与宽
        float origin_h = (float)origin_shape.get(0), origin_w = (float)origin_shape.get(1);

        // 计算x、y的缩放比例
        float im_scale_y, im_scale_x;
        if (keep_ratio) {
            float im_size_min = Math.min(origin_h, origin_w);
            float im_size_max = Math.max(origin_h, origin_w);
            int target_size_min = Math.min(target_size[0], target_size[1]);
            int target_size_max = Math.min(target_size[0], target_size[1]);
            float im_scale = target_size_min / im_size_min;
            if (Math.round(im_scale * im_size_max) > target_size_max) {
                im_scale = target_size_max / im_size_max;
            }
            im_scale_y = im_scale;
            im_scale_x = im_scale;

            target_size[0] = Math.round(im_scale_y * origin_h);
            target_size[1] = Math.round(im_scale_x * origin_w);
        } else {
            im_scale_y = target_size[0] / origin_h;
            im_scale_x = target_size[1] / origin_w;
        }

        // 图像缩放
        im = NDImageUtils.resize(im, target_size[1], target_size[0], interp);

        // 图像缩放比例
        NDArray scale_factor = manager.create(new float[] {im_scale_y, im_scale_x}).toType(DataType.FLOAT32, false);
        scale_factor = scale_factor.expandDims(0);

        // 返回Result
        return Lists.newArrayList(im, scale_factor);
    }

    /**
     * 图像归一化
     *
     * @param im            image ndarray
     * @param manager       NDArray Manager
     * @return NDArray
     */
    public static NDArray normalizeImage(NDArray im, NDManager manager) {
        return normalizeImage(im, null, null, true, "none", manager);
    }

    /**
     * 图像归一化
     *
     * @param im            image ndarray
     * @param mean          purpose: im - mean
     * @param std           purpose: im / std
     * @param is_scale      whether need im / 255
     * @param norm_type     type in ['mean_std', 'none']
     * @param manager       NDArray Manager
     * @return NDArray
     */
    public static NDArray normalizeImage(NDArray im, float[] mean, float[] std, boolean is_scale, String norm_type, NDManager manager) {
        if (!im.getDataType().equals(DataType.FLOAT32)) {
            im = im.toType(DataType.FLOAT32, false);
        }

        if (is_scale) {
            im = im.div(255f);
        }

        if ("mean_std".equals(norm_type)) {
            NDArray nmean = manager.create(mean, new Shape(1, 1, 3));
            NDArray nstd = manager.create(std, new Shape(1, 1, 3));
            im = im.sub(nmean);
            im = im.div(nstd);
        }

        return im;
    }

    /**
     * 图像转置（HWC ---> CHW）
     *
     * @param im        image ndarray
     * @param manager   NDArray Manager
     * @return NDArray
     */
    public static NDArray permute(NDArray im, NDManager manager) {
        // HWC ---> CHW
        im = im.transpose(2, 0, 1);
        im = im.expandDims(0);
        return im;
    }
}