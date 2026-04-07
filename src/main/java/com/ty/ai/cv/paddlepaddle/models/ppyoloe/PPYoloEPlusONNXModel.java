package com.ty.ai.cv.paddlepaddle.models.ppyoloe;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.onnxruntime.engine.OrtEngine;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import com.google.common.collect.Lists;
import com.ty.ai.cv.paddlepaddle.models.PaddleBaseModel;
import com.ty.ai.cv.paddlepaddle.translator.PPYoloETranslator;
import lombok.extern.slf4j.Slf4j;
import org.yaml.snakeyaml.Yaml;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * Paddle YoloE Plus ONNX Model
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
@Slf4j
public class PPYoloEPlusONNXModel implements PaddleBaseModel {

    private final String modelPath;
    private final String modelName = "model";
    private String cfgName = "cfg.yml";

    private ZooModel<Image, DetectedObjects> model;
    private Predictor<Image, DetectedObjects> predictor;

    private ImageFactory imageFactory = new BufferedImageFactory();

    public PPYoloEPlusONNXModel(String modelPath) {
        this.modelPath = modelPath;
        this.cfgName = modelPath + File.separator + this.cfgName;
    }

    /**
     * 模型初始化
     *
     * @return PPYoloEPlusModel
     * @throws Exception
     */
    public PPYoloEPlusONNXModel initialize() throws Exception {
        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optEngine(OrtEngine.ENGINE_NAME)
                        // .optOption("ortDevice", "TensorRT") // ONNX 1.18以上 支持CUDA12
                        .setTypes(Image.class, DetectedObjects.class)
                        .optModelPath(Paths.get(this.modelPath))
                        .optModelName(this.modelName)
                        .optTranslator(new PPYoloEPlusONNXTranslator(this.cfgName))
                        .optProgress(new ProgressBar())
                        .build();
        model = criteria.loadModel();
        predictor = model.newPredictor();
        log.info("PPYoloE Plus ONNX模型加载完毕 From {}/{}.onnx", this.modelPath, this.modelName);
        return this;
    }

    /**
     * Predicts image for inference.
     *
     * @param image 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    @Override
    public DetectedObjects predict(Image image) throws Exception {
        return predictor.predict(image);
    }

    /**
     * Predicts image for inference.
     *
     * @param bufferedImage 图片对象
     * @return DetectedObjects
     * @throws Exception
     */
    public DetectedObjects predict(BufferedImage bufferedImage) throws Exception {
        return this.predict(imageFactory.fromImage(bufferedImage));
    }

    /**
     * 释放模型使用的资源
     */
    @Override
    public void close() {
        if (null != predictor) {
            predictor.close();
            predictor = null;
        }
        if (null != model) {
            model.close();
            model = null;
        }
    }

    /**
     * Paddle YoloE Plus ONNX Translator
     */
    private static final class PPYoloEPlusONNXTranslator extends PPYoloETranslator {

        public PPYoloEPlusONNXTranslator(String cfgPath) {
            super(640, 640, false);
            super.setLabels(this.readLabels(cfgPath));
        }

        /**
         * 读取模型所有的Label
         */
        @SuppressWarnings("unchecked")
        List<String> readLabels(String cfgPath) {
            List<String> label_list = Lists.newArrayList();
            try {
                Yaml yaml = new Yaml();
                Map<String, Object> data = yaml.load(new FileReader(cfgPath));
                label_list = (List<String>) data.get("label_list");
            } catch (Exception e) {
                log.error(e.getMessage(), e);
            }
            return label_list;
        }
    }
}
