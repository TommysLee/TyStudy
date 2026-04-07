package com.ty.test.ai.cv;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import com.ty.ai.cv.paddlepaddle.models.scene.PedestrianDetectionModel;
import com.ty.utils.CVUtil;
import com.ty.utils.PathRootUtil;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Path;

/**
 * Pedestrian Detection Model Test / YOLOv3 Dacknet53 Model Test
 *
 * @Author Tommy
 * @Date 2024/5/5
 */
@Slf4j
public class PedestrianDetectionModelTest {

    static String imagePath = "src/main/resources/test/pedestrian_001.png";

    static void t1() throws Exception {
        Path imageFile = PathRootUtil.resolve(imagePath);
        Image image = ImageFactory.getInstance().fromFile(imageFile);

        PedestrianDetectionModel model = new PedestrianDetectionModel("/opt/paddle_models/pedestrian").initialize();

        log.info("开始推理：{}", imageFile);

        DetectedObjects detectedObjects = model.predict(image);

        log.info("推理结束 {}", imageFile);

        CVUtil.showResult(imageFile.toString(), detectedObjects);

        // 释放资源
        model.close();
    }

    public static void main(String[] args) throws Exception {
        t1();

        System.exit(0);
    }
}
