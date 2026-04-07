package com.ty.ai.cv.paddlepaddle.models.ppyoloe;

import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.DetectedObjects.DetectedObject;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import com.ty.ai.cv.paddlepaddle.models.PaddleBaseModel;
import com.ty.utils.CVUtil;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.Semaphore;

/**
 * Paddle YoloE Plus Model
 *
 * @Author Tommy
 * @Date 2024/5/3
 */
@Data
@Slf4j
public class PPYoloEPlusModel implements PaddleBaseModel {

    private ZooModel<Input, Output> model;

    private Predictor<Input, Output> predictor;

    private NDManager manager;

    private PPYoloEPlusTranslator translator;

    private String modelPath;

    private Integer gpuDeviceId;

    private Double threshold = 0.5d;

    private String pythonModelPath = "models/ppyoloEplus";

    private Semaphore lock = new Semaphore(1);

    /**
     * 实例化 Paddle YoloE Plus Model 对象
     *
     * @param modelPath Path of PaddleClas model
     */
    public PPYoloEPlusModel(String modelPath) {
        this.modelPath = modelPath;
    }

    /**
     * 实例化 Paddle YoloE Plus Model 对象
     *
     * @param modelPath Path of PaddleClas model
     * @param threshold 阈值 - 目标检测成功的阈值
     */
    public PPYoloEPlusModel(String modelPath, Double threshold) {
        this.modelPath = modelPath;
        this.threshold = threshold;
    }

    /**
     * 实例化 Paddle YoloE Plus Model 对象
     *
     * @param modelPath     Path of PaddleClas model
     * @param gpuDeviceId   Define which GPU card used to run model
     */
    public PPYoloEPlusModel(String modelPath, Integer gpuDeviceId) {
        this.modelPath = modelPath;
        this.gpuDeviceId = gpuDeviceId;
    }

    /**
     * 实例化 Paddle YoloE Plus Model 对象
     *
     * @param modelPath     Path of PaddleClas model
     * @param gpuDeviceId   Define which GPU card used to run model
     * @param threshold     阈值 - 目标检测成功的阈值
     */
    public PPYoloEPlusModel(String modelPath, Integer gpuDeviceId, Double threshold) {
        this.modelPath = modelPath;
        this.gpuDeviceId = gpuDeviceId;
        this.threshold = threshold;
    }

    /**
     * 模型初始化
     *
     * @return PPYoloEPlusModel
     */
    @Override
    public PPYoloEPlusModel initialize() throws Exception {
        Path modelPath = null;
        if (this.pythonModelPath.startsWith("models")) {
            modelPath = Paths.get(PPYoloEPlusModel.class.getClassLoader().getResource(this.pythonModelPath).toURI());
        } else {
            modelPath = Paths.get(this.pythonModelPath);
        }

        Criteria<Input, Output> criteria = Criteria.builder()
                .setTypes(Input.class, Output.class)
                .optModelPath(modelPath)
                .optEngine("Python")
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
        this.manager = NDManager.newBaseManager();
        this.translator = new PPYoloEPlusTranslator();

        log.info("AI模型配置信息 AI_MODEL={} DJL_Python_Interface={}", modelPath, pythonModelPath);
        return this;
    }

    /**
     * Predicts an item for inference.
     *
     * @param imagePath 图片路径
     * @return DetectedObjects
     * @throws Exception
     */
    @Override
    public DetectedObjects predict(String imagePath) throws Exception {
        if (StringUtils.isNoneBlank(imagePath)) {
            Input inputs = buildInput(imagePath,null);
            return this.predict(inputs);
        }
        return null;
    }

    /**
     * Predicts an item for inference.
     *
     * @param image 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    @Override
    public DetectedObjects predict(BufferedImage image) throws Exception {
        if (null != image) {
            Input inputs = buildInput(null, image);
            return this.predict(inputs);
        }
        return null;
    }

    /**
     * Predicts an item for inference.
     *
     * @param inputs Python Input Object
     * @return DetectedObjects
     * @throws Exception
     */
    private synchronized DetectedObjects predict(Input inputs) throws Exception {
        try {
            lock.acquire();
            Output outputs = this.predictor.predict(inputs);
            return this.translator.processOutput(this.manager, outputs, this.threshold);
        } finally {
            lock.release();
        }
    }

    /**
     * 筛选出Person的结果集
     *
     * @param detectedObjects DetectedObjects
     * @return List<DetectedObject>
     */
    public List<DetectedObject> filter_person(DetectedObjects detectedObjects) {
        return this.filter(detectedObjects, "人");
    }

    /**
     * 释放模型使用的资源
     *
     */
    @Override
    public void close() {
        try {
            lock.acquire();
        } catch (Exception e) {
            log.error(e.getMessage(), e);
        }

        if (null != this.predictor) {
            this.predictor.close();
        }
        if (null != this.model) {
            this.model.close();
        }
        if (null != this.manager) {
            this.manager.close();
        }
    }

    /**
     * Build Input Object
     *
     * @param imagePath 图片路径
     * @return Input
     */
    private Input buildInput(String imagePath, BufferedImage image) {
        Input inputs = new Input();
        inputs.addProperty("model", this.modelPath);
        if (null != this.gpuDeviceId) {
            inputs.addProperty("gpu", this.gpuDeviceId.toString()); // GPU加速卡编号
        }

        if (StringUtils.isNotBlank(imagePath)) {
            inputs.add("image", imagePath);
        } else if (null != image) {
            inputs.add("image_base64", CVUtil.image2Base64(image));
        }
        return inputs;
    }
}
