package com.ty.test.ai.cv;

import ai.djl.modality.cv.BufferedImageFactory;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import com.google.common.collect.Lists;
import com.ty.ai.cv.paddlepaddle.models.ppyoloe.PPYoloEPlusONNXModel;
import com.ty.utils.CVUtil;
import com.ty.utils.OnnxUtil;
import com.ty.utils.PathRootUtil;
import lombok.extern.slf4j.Slf4j;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Paddle YoloE Plus ONNX Model Test Based On DJL
 *
 * @Author Tommy
 * @Date 2024/5/14
 */
@Slf4j
public class PPYoloEPlusONNXModelTest {

    static String modelPath = "/opt/onnx_models/ppyoloE_plus_l";

    static String image1Path = "src/main/resources/test/test1.jpg";
    static String image2Path = "src/main/resources/test/test2.jpg";

    static {
        ImageFactory.setImageFactory(new BufferedImageFactory());
        OnnxUtil.loadShared();
    }

    /**
     * 单线程测试
     */
    static void t1() throws Exception {
        Path imageFile = PathRootUtil.resolve(image1Path);
        Image image = ImageFactory.getInstance().fromFile(imageFile);

        PPYoloEPlusONNXModel model = new PPYoloEPlusONNXModel(modelPath).initialize();

        log.info("开始推理：{}", imageFile);
        long begin = System.currentTimeMillis();

        DetectedObjects detectedObjects = model.predict(image);

        long end = System.currentTimeMillis();
        log.info("推理结束 {} \t目标检测数量：{}", imageFile, detectedObjects.getNumberOfObjects());
        System.out.printf("推理耗时：%s%n", end - begin);

        log.info("{}", detectedObjects);

        CVUtil.showResult(imageFile.toString(), detectedObjects);

        // 释放资源
        model.close();
    }

    /**
     * 多线程测试
     */
    static void t2() throws Exception {
        // 模型初始化
        PPYoloEPlusONNXModel model = new PPYoloEPlusONNXModel(modelPath).initialize();

        // 开始多线程
        int nThreads = 5;
        ExecutorService executorService = Executors.newFixedThreadPool(nThreads);
        List<Callable<Long>> tasks = Lists.newArrayListWithCapacity(nThreads);
        for (int i = 0; i < nThreads; i++) {
            tasks.add(() -> {
                long threadId = Thread.currentThread().getId();

                Path imageFile = PathRootUtil.resolve((threadId % 2 == 0? image1Path : image2Path));
                Image image = ImageFactory.getInstance().fromFile(imageFile);

                log.info("开始推理：{}", imageFile);

                DetectedObjects detectedObjects = model.predict(image);

                log.info("推理完成： {}", imageFile);
                return threadId;
            });
        }

        long begin = System.currentTimeMillis();
        List<Future<Long>> futures = executorService.invokeAll(tasks);
        for (Future<Long> f : futures) {
            log.info("Thread-{} Completed", f.get());
        }
        long end = System.currentTimeMillis();

        log.info("All Completed! 总耗时：{}ms", (end - begin));

        // 释放资源
        model.close();
        System.exit(0);
    }

    /**
     * 多图片测试
     */
    static void t3() throws Exception {
        Properties pps = System.getProperties();
        String cur_dir = pps.get("user.dir").toString() + File.separator;
        System.out.println("图片搜索目录：" + cur_dir);

        PPYoloEPlusONNXModel model = new PPYoloEPlusONNXModel(modelPath).initialize();

        // 识别test1 ~ test10图片
        for (int i = 1; i <= 50; i++) {
            File file = new File(cur_dir + "test" + i + ".jpg");
            if (!file.exists()) {
                continue;
            }

            Path imageFile = Paths.get(file.getPath());
            Image image = ImageFactory.getInstance().fromFile(imageFile);

            log.info("开始推理：{}", imageFile);

            long begin = System.currentTimeMillis();
            DetectedObjects detectedObjects = model.predict(image);
            long end = System.currentTimeMillis();

            log.info("推理结束 {} \t目标检测数量：{}", imageFile, detectedObjects.getNumberOfObjects());
            log.info("{}", detectedObjects);
            System.out.printf("%s 推理耗时：%s%n", file.getName(), end - begin);
            System.out.println();
            CVUtil.showResult(imageFile.toString(), detectedObjects);
        }
    }

    /*
     * 测试
     */
    public static void main(String[] args) throws Exception {
        t3();
    }
}
