package com.ty.ai.cv.paddlepaddle.models.scene;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.paddlepaddle.engine.PpEngine;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.ty.ai.cv.paddlepaddle.models.PaddleBaseModel;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Paths;

/**
 * 行人检测（Pedestrian Detection）
 *
 * 行人检测的主要应用有智能监控。在监控场景中，大多是从公共区域的监控摄像头视角拍摄行人，获取图像后再进行行人检测。
 *
 * https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.7/configs/pphuman/pedestrian_yolov3/README_cn.md
 *
 * @Author Tommy
 * @Date 2024/5/5
 */
@Data
@Slf4j
public class PedestrianDetectionModel implements PaddleBaseModel {

    private ZooModel<Image, DetectedObjects> model;

    private Predictor<Image, DetectedObjects> predictor;

    private String modelPath;

    /**
     * 实例化 Model 对象
     *
     * @param modelPath Path of PaddleClas model
     */
    public PedestrianDetectionModel(String modelPath) {
        this.modelPath = modelPath;
    }

    /**
     * 模型初始化
     *
     * @return PedestrianDetectionModel
     */
    @Override
    public PedestrianDetectionModel initialize() throws Exception {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optEngine(PpEngine.ENGINE_NAME)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelPath(Paths.get(this.modelPath))
                        .optModelName("inference")
                        .optTranslator(new PedestrianDetectionTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        model = criteria.loadModel();
        predictor = model.newPredictor();

        log.info("行人检测（Pedestrian Detection）加载完毕 {}", model.getModelPath());
        return this;
    }

    /**
     * Predicts an item for inference.
     *
     * @param image 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    @Override
    public DetectedObjects predict(Image image) throws Exception {
        return this.predictor.predict(image);
    }

    /**
     * 释放模型使用的资源
     *
     */
    @Override
    public void close() {
        if (null != this.predictor) {
            this.predictor.close();
        }
        if (null != this.model) {
            this.model.close();
        }
    }
}
