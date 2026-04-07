package com.ty.ai.cv.paddlepaddle.models;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import com.google.common.collect.Lists;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Paddle Base Model SDK
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
public interface PaddleBaseModel {

    /**
     * 模型初始化
     *
     * @return PaddleBaseModel
     * @throws Exception
     */
    PaddleBaseModel initialize() throws Exception;

    /**
     * Predicts image for inference.
     *
     * @param imagePath 图片路径
     * @return DetectedObjects
     * @throws Exception
     */
    default DetectedObjects predict(String imagePath) throws Exception {
        return null;
    }

    /**
     * Predicts image for inference.
     *
     * @param bufferedImage 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    default DetectedObjects predict(BufferedImage bufferedImage) throws Exception {
        return null;
    }

    /**
     * Predicts image for inference.
     *
     * @param image 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    default DetectedObjects predict(Image image) throws Exception {
        return null;
    }

    /**
     * 筛选指定标签的结果集
     *
     * @param detectedObjects   DetectedObjects
     * @param label             Label
     * @return List<DetectedObject>
     */
    default List<DetectedObject> filter(DetectedObjects detectedObjects, String label) {
        List<DetectedObject> detectedObjectList = Lists.newArrayList();
        for (int i = 0; i < detectedObjects.getNumberOfObjects(); i++) {
            DetectedObject item = detectedObjects.item(i);
            if (label.equals(item.getClassName())) {
                detectedObjectList.add(item);
            }
        }
        return detectedObjectList;
    }

    /**
     * 释放模型使用的资源
     */
    void close();
}
