package com.ty.test.paddle;

import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import com.ty.test.python.PythonTest;
import lombok.extern.slf4j.Slf4j;

import java.nio.file.Paths;

/**
 * Paddle YoloE Plus Model Test based on Python Engine 原生示例
 *
 * @Author Tommy
 * @Date 2024/5/1
 */
@Slf4j
public class PPYoloEPlusTest {

    static void t1(String imgPath) throws Exception {
        Criteria<Input, Output> criteria = Criteria.builder()
                .setTypes(Input.class, Output.class)
                .optModelPath(Paths.get(PythonTest.class.getClassLoader().getResource("models/ppyoloEplus2").toURI()))
                .optEngine("Python")
                .build();

        try(ZooModel<Input, Output> model = criteria.loadModel();
            Predictor<Input, Output> predictor = model.newPredictor()) {
            log.info("PPYoloE Plus模型加载完毕");

            Input inputs = new Input();
            inputs.addProperty("model", "D:/opt/models/PP-YOLOE-plus");
            inputs.addProperty("gpu", "0");
            inputs.add("image", imgPath);

            Output outputs = predictor.predict(inputs);
            System.out.println("这里：");
            System.out.println(outputs);

            NDManager manager = NDManager.newBaseManager();

            String scores = outputs.getAsString("scores");

            System.out.println(outputs.getAsString("boxes"));
            System.out.println(outputs.getAsString("labels"));
            System.out.println(scores);

            System.out.println("\r\n华丽的分割线-----------\r\n");
            System.out.println(manager);

            NDArray ndArray = manager.create(outputs.getAsString("boxes"));
            System.out.println(ndArray);

            // 仅用于测试的Code，可以忽略
            log.info("推理下一张图片....");

            inputs.add("image", "C:/Users/Honey/Desktop/Paddle/PP-YOLOE-L/example/test2.jpg");
            outputs = predictor.predict(inputs);
            log.info("outputs = {}", outputs);

            // DetectedObjects detectedObjects = predictor.predict(inputs);
            log.info("推理完成");
        }
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        t1("C:/Users/Honey/Desktop/Paddle/PP-YOLOE-L/example/test1.jpg");
    }
}
